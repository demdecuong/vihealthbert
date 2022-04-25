import torch
import torch.nn as nn
from .module import Classifier
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


class ViHnBERT(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(ViHnBERT, self).__init__(config)
        self.args = args
        self.config = config
        # init backbone
        self.roberta = RobertaModel(config)

        self.classifier = Classifier(config, args.dropout_rate)

    
    def forward(self,
                input_ids=None, 
                token_type_ids=None, 
                attention_mask=None, 
                start_token_idx=None, 
                end_token_idx=None,
                labels=None):

        outputs = self.roberta(input_ids=input_ids,
                                    attention_mask=attention_mask)

        features_bert = outputs[0]
        # Features of [CLS] tokens
        features_cls = features_bert[:, 0, :].unsqueeze(1)

        # Features of acronym tokens
        if start_token_idx is None or end_token_idx is None:
            raise Exception('Require start_token_idx and end_token_idx')
        list_mean_feature_acr = []
        for idx in range(features_bert.size()[0]):
            feature_acr = features_bert[idx, start_token_idx[idx]:end_token_idx[idx]+1, :].unsqueeze(0)
            mean_feature_acr = torch.mean(feature_acr, 1, True)
            list_mean_feature_acr.append(mean_feature_acr)
        features_arc = torch.cat(list_mean_feature_acr, dim=0)

        # Concate featrues
        features = torch.cat([features_cls, features_arc], dim=2)

        logits = self.classifier(features)
        outputs = ((logits),) + outputs[2:]

        loss_fn = nn.BCELoss()
        total_loss = 0.0

        if labels is not None:
            total_loss = loss_fn(logits, labels)
        
        outputs = (total_loss,) + outputs
        
        return outputs



