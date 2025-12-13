import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel

import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def set_validator(self, validator):
        self.validator = validator

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, return_loss=None):
        if isinstance(inputs, dict) is False:
            query, passage = inputs
            return model(query=query, passage=passage).loss
        else:
            query, passage = inputs['inputs'] # Hacky workaround for `prediction_step`
            outputs = model(query=query, passage=passage)
            loss = outputs.loss
            self.log({f"eval/{key}": outputs.logs[key] for key in outputs.logs})

            return (loss, [])

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        Returns the correct autocast context depending on CPU/GPU AMP settings.
        Supports bf16 autocast on GPU.
        """

        dtype = torch.float32
        dtype = torch.bfloat16 if self.args.bf16 else dtype
        dtype = torch.fploat16 if self.args.fp16 else dtype

        # CPU autocast
        if self.use_cpu_amp:
            return torch.autocast(
                device_type="cpu",
                dtype=dtype,      # e.g., torch.bfloat16
                cache_enabled=cache_enabled
            )

        # GPU autocast (bf16 or fp16 depending on self.amp_dtype)
        if torch.cuda.is_available():
            return torch.autocast(
                device_type="cuda",
                dtype=dtype,      # important: bf16 works only on Ampere+
                cache_enabled=cache_enabled
            )

        # otherwise no autocast
        return contextlib.nullcontext()

    def training_step(self, *args):
        if self.state.global_step % self.args.eval_steps == 0: 
            self.prediction_step(*args)
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    # def prediction_step(self, models, inputs, *args, **kwargs):
    #     query, passage = inputs
    #     inputs = {'inputs': inputs, 'return_loss': True}
    #     return super(TevatronTrainer, self).prediction_step(models, inputs, *args, **kwargs)

    # NOTE: move to two device?
    def prediction_step(self, models, inputs, *args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            with torch.no_grad():
                logs = self.validator.run(models, self.args.device)
                sum_values = 0.0
                for key, value in logs.items():
                    self.log({f"eval/{key}": value})
                    sum_values += value

                self.log({"eval/avg": float(sum_values / len(logs))})
                avg = torch.tensor(sum_values / len(logs), device=self.args.device)
            return (avg, None, None)
        else:
            return (None, None, None)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        data_collator = self.data_collator
        return DataLoader(
            eval_dataset,
            sampler=None,
            collate_fn=data_collator,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False
        )

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
