import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from tevatron.retriever.trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)

# TODO: see if we can use all the in-batch negative when detaching the teacher scores.
# TODO: remove the debugging print out
class TevatronCovDistilTrainer(TevatronTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, return_loss=None):
        if isinstance(inputs, dict) is False:
            query, passage, subquery, num_subqueries = inputs
            num_subqueries = torch.tensor(num_subqueries, device=query['input_ids'].device)

            # get shape/sizes
            group_size = passage['input_ids'].size(0) // query['input_ids'].size(0)
            local_batch_size = query['input_ids'].size(0)

            # Loss1 -- loss_rel: standard forward passing (already gathered scores and reps)
            output = model(query=query, passage=passage)
            loss_rel, student_scores = output.loss, output.scores
            p_reps = output.p_reps
            batch_size = student_scores.size(0)

            if hasattr(model, 'module'):
                sq_reps = model.module.encode_query(subquery)
                temperature = model.module.temperature
            else:
                sq_reps = model.encode_query(subquery)
                temperature = model.temperature

            # NOTE: each rank is going to compute the subquery relevance they have and aggregate them to the specific query
            teacher_scores_local = torch.ones(local_batch_size, p_reps.size(0), device=sq_reps.device)
            teacher_scores_local_sq = torch.matmul(sq_reps, p_reps.transpose(0, 1))
            sq_offsets = torch.cumsum(torch.cat([num_subqueries.new_zeros(1), num_subqueries]), dim=0)

            assert max(sq_offsets)==teacher_scores_local_sq.size(0), 'Mismatched sizes'
            for idx in range(local_batch_size):
                sq_start_idx = sq_offsets[idx]
                sq_end_idx   = sq_offsets[idx+1]
                scores = teacher_scores_local_sq[sq_start_idx: sq_end_idx]
                if self.args.aggregation_strategy=='sum':
                    teacher_scores_local[idx] = torch.sum(scores, dim=0)
                if self.args.aggregation_strategy=='mean':
                    teacher_scores_local[idx] = torch.mean(scores, dim=0)

            # gather the teacher score 
            if hasattr(model, 'module'):
                if model.module.is_ddp:
                    teacher_scores = model.module._dist_gather_tensor(teacher_scores_local)
            else:
                if model.is_ddp:
                    teacher_scores = model._dist_gather_tensor(teacher_scores_local)

            #### sanity check
            if self.state.global_step % 100 == 0:
                k = 10
                s_topk = student_scores.topk(k, dim=1).indices
                t_topk = teacher_scores.topk(k, dim=1).indices
                overlap = (s_topk == t_topk).float().mean().item()
                self.log({"overlap": overlap})

                s_probs = torch.softmax(student_scores.float(), dim=1)
                t_probs = torch.softmax(teacher_scores.float(), dim=1)
                s_ent = -(s_probs * torch.log(s_probs + 1e-9)).sum(dim=1).mean()
                t_ent = -(t_probs * torch.log(t_probs + 1e-9)).sum(dim=1).mean()
                self.log({"student entropy:": s_ent.item()})
                self.log({"teacher entropy:": t_ent.item()})

                print('student_scores (0)', student_scores[0, :16])
                print('teacher_scores (0)', teacher_scores[0, :16])
                print('student_scores (-1)', student_scores[-1, -16:])
                print('teacher_scores (-1)', teacher_scores[-1, -16:])

            ## Loss2: subquery_contrsative
            if self.args.subquery_constrastive:
                teacher_scores = teacher_scores.view(batch_size, -1)
                target = torch.arange(batch_size, device=teacher_scores.device, dtype=torch.long)
                target = target * group_size
                loss_subrel = F.cross_entropy(
                    teacher_scores / temperature, 
                    target,
                    reduction='mean'
                ) * self._dist_loss_scale_factor
            else:
                loss_subrel = torch.tensor(0.0)

            # Loss3: Resahpe the student without in-batch
            teacher_scores = teacher_scores.detach()  # detach for the KD part

            T = self.args.distil_temperature
            student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
            teacher_probs = torch.softmax(teacher_scores.float()    / T, dim=1)

            # KL Divergence loss (shapes now [batch, num_labels])
            loss_distil = torch.nn.functional.kl_div(
                student_log,
                teacher_probs,
                reduction="batchmean"
            ) * self._dist_loss_scale_factor

            # loss summation # NOTE: to also monitor the changes when lambda == 0, switch to 1. 
            # NOTE: but still use zero when calculating loss for FP/BP
            covdistil_lambda = self.args.covdistil_lambda if self.args.covdistil_lambda != 0 else 1
            self.log({
                "rel-constrast": loss_rel.item(), 
                "subrel-constrast": loss_subrel.item(),
                "cov-distil": loss_distil.item(),
            })
            loss = loss_rel + \
                   loss_subrel * self.args.covdistil_lambda,
                   loss_distil * self.args.covdistil_lambda
            return loss
        else:
            query, passage = inputs['inputs'] # Hacky workaround for `prediction_step`
            outputs = model(query=query, passage=passage)
            loss = outputs.loss
            self.log({f"eval/{key}": outputs.logs[key] for key in outputs.logs})

            return (loss, [])

# NOTE: comment out for now
# class DistilTevatronTrainer(TevatronTrainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.is_ddp = dist.is_initialized()
#         self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
# 
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
# 
#         if isinstance(inputs, dict) is False:
#             query, passage, reranker_labels = inputs
#             scores = model(query=query, passage=passage).scores
#             
#             # reranker_scores are gathered across all processes
#             if hasattr(model, 'module'):
#                 if model.module.is_ddp:
#                     reranker_labels = model.module._dist_gather_tensor(reranker_labels)
#             else:
#                 if model.is_ddp:
#                     reranker_labels = model._dist_gather_tensor(reranker_labels)
#             
#             # Derive student_scores [batch, num_labels]
#             batch_size, total_passages = scores.size()
#             num_labels = reranker_labels.size(1)
#             start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
#             idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
#             student_scores = scores.gather(1, idx_matrix)
# 
#             # Temperature‐scaled soft distributions
#             T = self.args.distil_temperature
#             student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
#             teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)
# 
#             # KL Divergence loss (shapes now [batch, num_labels])
#             loss = torch.nn.functional.kl_div(
#                 student_log,
#                 teacher_probs,
#                 reduction="batchmean"
#             ) * self._dist_loss_scale_factor
# 
#             return loss
# 
#         else:
#             query, passage, _ = inputs['inputs'] # Hacky workaround for `prediction_step`
#             outputs = model(query=query, passage=passage)
#             loss = outputs.loss
#             self.log({f"eval/{key}": outputs.logs[key] for key in outputs.logs})
# 
#             return (loss, [])
# 
#     def training_step(self, *args):
#         if self.state.global_step % self.args.eval_steps == 0: 
#             self.prediction_step(*args)
#         return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor
# 
#     # def prediction_step(self, models, inputs, *args, **kwargs):
#     #     query, passage, _ = inputs
#     #     inputs = {'inputs': inputs, 'return_loss': True}
#     #     return super(TevatronTrainer, self).prediction_step(models, inputs, *args, **kwargs)
