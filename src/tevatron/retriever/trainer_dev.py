import os
from typing import Optional
import wandb

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from tevatron.retriever.trainer import TevatronTrainer
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback

import logging
logger = logging.getLogger(__name__)

class TevatronCovDistilTrainer(TevatronTrainer):

    def compute_orthogonal_loss(self, views):
        if views.dim() != 3:
            return torch.tensor(0)

        B, V, H = views.shape
        sim = torch.bmm(views, views.transpose(1, 2))
        eye = torch.eye(V, device=views.device).unsqueeze(0)
        num_pairs = B * V * (V - 1)

        if self.args.view_orthogonalize_method == 'mse':
            loss = ((sim - eye) ** 2).sum() / num_pairs
        elif self.args.view_orthogonalize_method == 'abs':
            loss = torch.abs(sim * (1 - eye)).sum() / num_pairs
        else:
            sim = sim * (1 - eye)
            loss = (sim ** 2).sum() / num_pairs
        return loss * self._dist_loss_scale_factor

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
            q_reps = output.q_reps # this will be multi-view
            batch_size = student_scores.size(0)

            # Loss4 -- Orthogonal loss
            loss_orthogonal = self.compute_orthogonal_loss(q_reps)

            if hasattr(model, 'module'):
                sq_reps = model.module.encode_query(subquery)
                temperature = model.module.temperature
            else:
                sq_reps = model.encode_query(subquery)
                temperature = model.temperature

            # NOTE: each rank is going to compute the subquery relevance they have and aggregate them to the specific query
            teacher_scores_local = torch.zeros(local_batch_size, p_reps.size(0), device=sq_reps.device)
            teacher_scores_local_sq = torch.matmul(sq_reps, p_reps.transpose(0, 1))
            sq_offsets = torch.cumsum(torch.cat([num_subqueries.new_zeros(1), num_subqueries]), dim=0)

            assert max(sq_offsets)==teacher_scores_local_sq.size(0), 'Mismatched sizes'
            for idx in range(local_batch_size):
                sq_start_idx = sq_offsets[idx]
                sq_end_idx   = sq_offsets[idx+1]
                scores = teacher_scores_local_sq[sq_start_idx: sq_end_idx]
                if self.args.aggregation_strategy=='max':
                    teacher_scores_local[idx] = torch.max(scores, dim=0).values
                else:
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
                # view similarity check 
                if q_reps.dim() == 3:
                    q_reps_ = q_reps.detach()
                    sim_mat = torch.bmm(q_reps_, q_reps_.transpose(1, 2))  # B, V, V
                    sim_mat = sim_mat.mean(dim=0).to(torch.float32).cpu().numpy()
                    np.fill_diagonal(sim_mat, np.nan)

                    fig, ax = plt.subplots(figsize=(4, 4))  
                    im = ax.imshow(sim_mat, cmap="coolwarm")
                    fig.colorbar(im, ax=ax)

                    for cb in self.callback_handler.callbacks:
                        if isinstance(cb, WandbCallback):
                            control = cb.on_log(self.args, self.state, self.control, logs={"views_heatmap": wandb.Image(fig)})
                    plt.close(fig)

                # teahcer-student overalp check
                k = 10
                s_topk = student_scores.topk(k, dim=1).indices
                t_topk = teacher_scores.topk(k, dim=1).indices
                overlap = (s_topk == t_topk).float().mean().item()
                self.log({"overlap": overlap})

                # margin checks
                scores_3d = student_scores.view(batch_size, batch_size, -1)
                scores = torch.diagonal(scores_3d, dim1=0, dim2=1).permute(1, 0)
                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1:].max(1).values
                margin_s = pos_scores - neg_scores
                self.log({"student margin": margin_s.mean().item()})

                scores_3d = teacher_scores.view(batch_size, batch_size, -1)
                scores = torch.diagonal(scores_3d, dim1=0, dim2=1).permute(1, 0)
                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1:].max(1).values
                margin_t = pos_scores - neg_scores
                self.log({"teacher margin": margin_t.mean().item()})
                self.log({"margin difference": (margin_t - margin_s).mean().item()})

            ## Loss2: subquery_contrsative
            teacher_scores = teacher_scores.view(batch_size, -1)
            target = torch.arange(batch_size, device=teacher_scores.device, dtype=torch.long)
            target = target * group_size
            loss_subrel = F.cross_entropy(
                teacher_scores / temperature, 
                target,
                reduction='mean'
            ) * self._dist_loss_scale_factor

            # Loss3 
            # NOTE old setting considers all the in-batch negative for distillation (deprecated)
            # NOTE: the new setting considers only own-negative for distillation (like the normal KD)
            if self.args.covdistil_method == 'KLD':
                T = self.args.distil_temperature
                student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
                teacher_probs = torch.softmax(teacher_scores.detach().float()    / T, dim=1)
                loss_distil = torch.nn.functional.kl_div(
                    student_log, teacher_probs,
                    reduction="batchmean"
                ) * self._dist_loss_scale_factor

            elif self.args.covdistil_method == 'MarginMSE':
                start_idx = torch.arange(0, batch_size * group_size, group_size, device=student_scores.device)
                idx_matrix = start_idx.view(-1, 1) + torch.arange(group_size, device=student_scores.device)
                student_scores_group = student_scores.gather(1, idx_matrix)
                teacher_scores_group = teacher_scores.detach().gather(1, idx_matrix)

                student_margin = student_scores_group[:, 0:1] - student_scores_group[:, 1:]
                teacher_margin = teacher_scores_group[:, 0:1] - teacher_scores_group[:, 1:]
                loss_distil = F.mse_loss(student_margin, teacher_margin) * self._dist_loss_scale_factor * 200 
                # NOTE: scale up to the level of KLD

            # loss summation # NOTE: to also monitor the changes when lambda == 0, switch to 1. 
            # NOTE: but still use zero when calculating loss for FP/BP
            self.log({
                "rel-constrast": loss_rel.item(), 
                "subrel-constrast": loss_subrel.item(),
                "cov-distil": loss_distil.item(),
                "view-similarity": loss_orthogonal.item(),
            })
            loss = loss_rel * self.args.contrastive_lambda
            loss += loss_subrel * self.args.sq_contrastive_lambda 
            loss += loss_distil * self.args.covdistil_lambda
            loss += loss_orthogonal * self.args.view_orthogonalize_lambda
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
