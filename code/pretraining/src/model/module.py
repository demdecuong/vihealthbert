import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from transformers.activations import gelu

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, labels, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(x.view(-1, self.config.vocab_size), labels.view(-1))

        return x, masked_lm_loss

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class RobertaOnlyNSPHead(nn.Module):
    """Roberta Head for Next Sentence Prediction"""

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output, labels):
        seq_relationship_score = self.seq_relationship(pooled_output)
      
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = F.binary_cross_entropy_with_logits(
                seq_relationship_score,
                labels.float(),
                reduction='none').mean()
            # next_sentence_loss = loss_fct(seq_relationship_score, labels.view(-1))

        return seq_relationship_score, next_sentence_loss

class RobertaCapitalPredHead(nn.Module):
    """Roberta Head for Capitalize Prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, 1)
        
    def forward(self,input_ids, sequence_output, labels_cap, args):
        x = self.dense(sequence_output)
        x = gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        cap_loss = None
        if labels_cap is not None:
            labels_cap = labels_cap.unsqueeze(2)
            cap_loss = F.binary_cross_entropy_with_logits(
                x,
                labels_cap.float(),
                reduction='none').squeeze(2)
            
            mask = (input_ids==args.pad_token_id)
            valid_mask = ~mask
            cap_len = valid_mask.float().sum(dim=1)
            cap_loss.masked_fill_(mask, 0)
            cap_loss = cap_loss.sum(dim=1) / cap_len
            cap_loss = cap_loss.mean()

        return x, cap_loss

class RobertaSDHead(nn.Module):
    '''
    Roberta Head for sentence-distance task that indentify relationship between 2 sentences
        0: 2 sentences are adjacent doc
        1: _______________ same _______
        2: _______________ different _
    '''
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 3) 

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score



