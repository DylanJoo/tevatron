import torch
import logging
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def encode_query(self, qry, num_views=0, ind_pooling=False):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if num_views == 0:
            return self._pooling(query_hidden_states, qry['attention_mask'])

        masked_hiddens = query_hidden_states.masked_fill(~qry['attention_mask'][..., None].bool(), 0.0)
        query_mask = qry['attention_mask'].clone()

        ## NOTE: View tokens in the begining. 
        ## Tempalte: [CLS]search_query: [unused0]....[unusedn] {q}[SEP][PAD]...
        ## Tempalte: [PAD]...[CLS]search_query: {q}[SEP][unused0]....[unusedn][SEP]

        ### NOTE: incremental pooling
        if ind_pooling:
            query_sum = (masked_hiddens * query_mask[..., None]).sum(dim=1)
            query_len = query_mask.sum(dim=1)
            views_sum = masked_hiddens[:, 5:(5+num_views), :] # B 5 H
            views_counts = query_len[..., None] + torch.ones(num_views, device=masked_hiddens.device)[None, ...] # B 5
            views_reps = query_sum.unsqueeze(1) + views_sum # B 5 H
            reps = views_reps / views_counts.unsqueeze(-1)
        else:
            query_mask[:, 5:(5+num_views)] = False # exclude view tokens
            query_sum = (masked_hiddens * query_mask[..., None]).sum(dim=1)
            query_len = query_mask.sum(dim=1)
            views_sum = (masked_hiddens[:, 5:(5+num_views), :]).cumsum(dim=1) # B 5 H
            views_counts = query_len[..., None] + torch.arange(1, num_views + 1, device=masked_hiddens.device)[None, ...] # B 5
            views_reps = query_sum.unsqueeze(1) + views_sum # B 5 H
            reps = views_reps / views_counts.unsqueeze(-1)

        # query_mask[:, -(num_views+1):] = False
        # query_reps = (masked_hiddens * query_mask[..., None]).sum(dim=1) / query_mask.sum(dim=1)[..., None]
        # views_reps = masked_hiddens[:, -(num_views+1):]

        # reps = torch.cat([query_reps.unsqueeze(1), views_reps], dim=1)
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


class MultiModalDenseModel(DenseModel):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(self, encoder, pooling='eos', normalize=True, temperature=0.02):
        super().__init__(encoder, pooling, normalize, temperature)
        # freeze visual encoder
        self.encoder = encoder
        for param in self.encoder.visual.parameters():
            param.requires_grad = False
        # freeze audio_tower
        for param in self.encoder.audio_tower.parameters():
            param.requires_grad = False
        self.config.hidden_size = 3584

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def encode_query(self, qry):
        cache_position = torch.arange(0, qry['input_ids'].shape[1], device=qry['input_ids'].device)
        qry = self.encoder.prepare_inputs_for_generation(**qry, use_cache=True, cache_position=cache_position)
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        # query_hidden_states = query_hidden_states.hidden_states[1][-1]
        query_hidden_states = query_hidden_states.hidden_states[-1]

        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
