from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from .module import RobertaLMHead, RobertaOnlyNSPHead, RobertaCapitalPredHead


class ViHnBERT(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(ViHnBERT, self).__init__(config)
        self.args = args
        self.config = config
        # init backbone
        self.roberta = RobertaModel(config)
        # init task layer 
        self.lm_head = RobertaLMHead(config)
        if args.do_nsp:
            self.nsp_clf = RobertaOnlyNSPHead(config)
        if args.do_cap:
            self.cap_clf = RobertaCapitalPredHead(config)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                labels_mlm=None,
                labels_nsp=None,
                labels_cap=None,
                
                ):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        masked_lm_loss = None
        masked_lm_scores, masked_lm_loss = self.lm_head(
                                                sequence_output,
                                                labels_mlm)
        
        nsp_loss = None
        if self.args.do_nsp:
            nsp_scores, nsp_loss = self.nsp_clf(
                                            pooled_output,
                                            labels_nsp)

        cap_loss = None
        if self.args.do_cap:
            cap_scores, cap_loss = self.cap_clf(
                                        input_ids,
                                        sequence_output,
                                        labels_cap,
                                        self.args) 

        # Combine scores and hidden state values
        pred_scores = (masked_lm_scores,)
        if self.args.do_nsp:
            pred_scores += (nsp_scores,)
        if self.args.do_cap:
            pred_scores += (cap_scores,)

        output = pred_scores + outputs[2:]

        # Return with all loss
        total_loss = 0
        mlm_coef = 1

        loss = ()
        if nsp_loss is not None:
            loss += (nsp_loss,)
            total_loss += self.args.nsp_coef * nsp_loss
            mlm_coef -= self.args.nsp_coef
        
        if cap_loss is not None:
            loss += (cap_loss,)
            total_loss += self.args.cap_coef * cap_loss
            mlm_coef -= self.args.cap_coef
        
        if masked_lm_loss is not None:
            loss = (masked_lm_loss,) + loss
            total_loss += mlm_coef * masked_lm_loss

        # total_loss , mlm_loss, cap_loss , nsp_loss, mlm_score, cap_score, nsp_score , hidden state,
        output = (total_loss,) + loss + output
        return output
