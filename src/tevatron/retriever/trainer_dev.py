import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from tevatron.retriever.trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)

class TevatronDualDistilTrainer(TevatronTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # the coverage scores (last element) are not used by the loss
        query, passage, subquery, num_subqueries, _ = inputs
        num_subqueries = torch.tensor(num_subqueries, device=query['input_ids'].device)

        # get shape/sizes
        group_size = passage['input_ids'].size(0) // query['input_ids'].size(0)
        local_batch_size = query['input_ids'].size(0)

        # Loss1 -- loss_contrastive_stu
        output = model(query=query, passage=passage)
        loss_contrastive_stu, student_scores = output.loss, output.scores
        p_reps, q_reps = output.p_reps, output.q_reps
        batch_size = student_scores.size(0)

        if hasattr(model, 'module'):
            sq_reps = model.module.encode_query(subquery)
            temperature = model.module.temperature
        else:
            sq_reps = model.encode_query(subquery)
            temperature = model.temperature

        # NOTE: each rank is going to compute the subquery relevance they have and aggregate them to the specific query
        teacher_scores_slocal = torch.zeros(local_batch_size, p_reps.size(0), device=sq_reps.device)
        teacher_scores_local_sq = torch.matmul(sq_reps, p_reps.transpose(0, 1))
        sq_offsets = torch.cumsum(torch.cat([num_subqueries.new_zeros(1), num_subqueries]), dim=0)

        assert max(sq_offsets)==teacher_scores_local_sq.size(0), 'Mismatched sizes'
        for idx in range(local_batch_size):
            sq_start_idx = sq_offsets[idx]
            sq_end_idx   = sq_offsets[idx+1]
            scores = teacher_scores_local_sq[sq_start_idx: sq_end_idx]
            # teacher_scores_slocal[idx] = torch.max(scores, dim=0).values
            teacher_scores_slocal[idx] = torch.mean(scores, dim=0)

        # gather the teacher score 
        if hasattr(model, 'module'):
            if model.module.is_ddp:
                teacher_scores = model.module._dist_gather_tensor(teacher_scores_slocal)
        else:
            if model.is_ddp:
                teacher_scores = model._dist_gather_tensor(teacher_scores_slocal)

        #### sanity check
        if self.state.global_step % 100 == 0:
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
        loss_contrastive_tea = F.cross_entropy(
            teacher_scores / temperature, 
            target,
            reduction='mean'
        ) * self._dist_loss_scale_factor

        ## Loss3 : loss_selfconvdistil
        T = self.args.distil_temperature
        student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
        teacher_probs = torch.softmax(teacher_scores.detach().float()    / T, dim=1)
        loss_selfconvdistil = torch.nn.functional.kl_div(
            student_log, teacher_probs,
            reduction="batchmean"
        ) * self._dist_loss_scale_factor

        # loss summation (all terms are logged, even when their lambda is 0)
        self.log({
            "rel-constrast": loss_contrastive_stu.item(), 
            "subrel-constrast": loss_contrastive_tea.item(),
            "cov-distil": loss_selfconvdistil.item(),
        })
        loss = loss_contrastive_stu * self.args.contrastive_lambda
        loss += loss_contrastive_tea * self.args.sq_contrastive_lambda
        loss += loss_selfconvdistil * self.args.covdistil_lambda
        return loss
