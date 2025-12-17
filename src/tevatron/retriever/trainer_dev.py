import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from tevatron.retriever.trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


class TevatronCovDistilTrainer(TevatronTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, return_loss=None):
        if isinstance(inputs, dict) is False:
            query, passage, subquery, num_subqueries = inputs

            # get shape/sizes
            group_size = passage['input_ids'].size(0) // query['input_ids'].size(0)
            # print('group_size', group_size)

            # standard forward passing (already gathered scores and reps)
            output = model(query=query, passage=passage)
            loss_rel, student_scores = output.loss, output.scores
            p_block_reps = output.p_reps.view(-1, group_size, output.p_reps.size(-1))
            batch_size = student_scores.size(0)
            # print('student_scores', student_scores.shape)
            # print('p_block_reps', p_block_reps.shape)
            # print('batch_size', batch_size)

            # forward subquery and calculate relevance individually
            if hasattr(model, 'module'):
                sq_reps = model.module.encode_query(subquery)
                if model.module.is_ddp:
                    sq_reps = model.module._dist_gather_tensor(sq_reps)
                    num_subqueries = model.module._dist_gather_tensor(num_subqueries)
            else:
                sq_reps = model.encode_query(subquery)
                if model.is_ddp:
                    sq_reps = model._dist_gather_tensor(sq_reps)
                    num_subqueries = model._dist_gather_tensor(num_subqueries)

            # print('sq_reps', sq_reps.shape)
            # print('num_subqueries', num_subqueries)
            # calcuate teacher scores
            teacher_scores = torch.full(
                (batch_size, group_size), -float("inf"), device=student_scores.device
            )

            offset = 0
            for idx in range(batch_size):
                n_sq = num_subqueries[idx]
                scores = sq_reps[offset: (offset+n_sq)] @ p_block_reps[idx].T # (m, h) x (n, h)
                # NOTE: aggregation strategy
                teacher_scores[idx] = torch.logsumexp(scores, dim=0) # (m, n) --> (n,)
                offset += n_sq

            # print('teacher_scores', teacher_scores.shape)
            # calculate local covdistillation
            student_scores_local = student_scores.view(batch_size, batch_size, group_size)
            student_scores_local = student_scores_local.diagonal(dim1=0, dim2=1).transpose(0, 1)
            # NOTE: (B, B, N)...in each query over B, it has B rows of scores. 
            # NOTE: In each row, it represents the distribution of all the groups if docs 
            # NOTE: to assign the "own" docs, we will get student_scores_local[i, i, :] 
            # NOTE: Finally, do the transpose (N, B) --> (B, N)
            # print('student_scores_local', student_scores_local.shape)

            T = self.args.distil_temperature
            student_log   = torch.log_softmax(student_scores_local.float() / T, dim=1)
            teacher_probs = torch.softmax(teacher_scores.float()    / T, dim=1)
            print('student_log', student_log[0])
            print('teacher_probs', teacher_probs[0])

            # KL Divergence loss (shapes now [batch, num_labels])
            loss_distil = torch.nn.functional.kl_div(
                student_log,
                teacher_probs,
                reduction="batchmean"
            ) * self._dist_loss_scale_factor

            # loss
            self.log({"train/constrastive": loss_rel, "train/covdistil": loss_distil})
            return loss_rel + loss_distil * 0.5
        else:
            query, passage = inputs['inputs'] # Hacky workaround for `prediction_step`
            outputs = model(query=query, passage=passage)
            loss = outputs.loss
            self.log({f"eval/{key}": outputs.logs[key] for key in outputs.logs})

            return (loss, [])

class DistilTevatronTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if isinstance(inputs, dict) is False:
            query, passage, reranker_labels = inputs
            scores = model(query=query, passage=passage).scores
            
            # reranker_scores are gathered across all processes
            if hasattr(model, 'module'):
                if model.module.is_ddp:
                    reranker_labels = model.module._dist_gather_tensor(reranker_labels)
            else:
                if model.is_ddp:
                    reranker_labels = model._dist_gather_tensor(reranker_labels)
            
            # Derive student_scores [batch, num_labels]
            batch_size, total_passages = scores.size()
            num_labels = reranker_labels.size(1)
            start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
            idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
            student_scores = scores.gather(1, idx_matrix)

            # Temperature‐scaled soft distributions
            T = self.args.distil_temperature
            student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
            teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)

            # KL Divergence loss (shapes now [batch, num_labels])
            loss = torch.nn.functional.kl_div(
                student_log,
                teacher_probs,
                reduction="batchmean"
            ) * self._dist_loss_scale_factor

            return loss

        else:
            query, passage, _ = inputs['inputs'] # Hacky workaround for `prediction_step`
            outputs = model(query=query, passage=passage)
            loss = outputs.loss
            self.log({f"eval/{key}": outputs.logs[key] for key in outputs.logs})

            return (loss, [])

    def training_step(self, *args):
        if self.state.global_step % self.args.eval_steps == 0: 
            self.prediction_step(*args)
        return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    # def prediction_step(self, models, inputs, *args, **kwargs):
    #     query, passage, _ = inputs
    #     inputs = {'inputs': inputs, 'return_loss': True}
    #     return super(TevatronTrainer, self).prediction_step(models, inputs, *args, **kwargs)
