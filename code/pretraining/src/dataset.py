from torch.utils.data import Dataset
import torch
import os
import random
import re

from utils import read_line_by_line


class ViHnDataset(Dataset):
    def __init__(self, args, tokenizer, mode) -> None:
        super().__init__()

        self.args = args

        DATA_PATH = os.path.join(self.args.data_dir, mode, self.args.data_name)

        self.data = read_line_by_line(DATA_PATH)
        
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)-1

    def __getitem__(self, index: int):
        text = self.data[index]
        while True:
            if not re.search('\w', text):
                text = self.data[random.randint(0, len(self.data)-1)]
            else:
                break
        if text == self.args.special_char_end_news:
            text = self.data[index-2]
            flags = False
        else:
            flags = True

        # Next Sentence Prediction
        if self.args.do_nsp:
            type_class = torch.bernoulli(torch.tensor(0.5)).bool()
            if type_class and flags:
                next_sentence = self.data[index+1]
                label_nsp = torch.tensor([1])
            elif type_class and not flags:
                next_sentence = self.data[index-1]
                label_nsp = torch.tensor([1])
            else:
                next_sentence = self.data[random.randint(0, len(self.data)-1)]
                label_nsp = torch.tensor([0])

        # Capitalize Prediction
        if self.args.do_cap:
            cap_labels = []
            for token in text.split(' '):
                if token[0].isupper():
                    cap_labels.extend(
                        [1] * len(self.tokenizer(token)['input_ids'][1:-1]))
                else:
                    cap_labels.extend(
                        [0] * len(self.tokenizer(token)['input_ids'][1:-1]))
            cap_labels = [0] + cap_labels + [0] # CLS , SEP
            # CLS and SEP token
            if self.args.do_nsp:
                for token in next_sentence.split(' '):
                    if token[0].isupper():
                        cap_labels.extend(
                            [1] * len(self.tokenizer(token)['input_ids'][1:-1]))
                    else:
                        cap_labels.extend(
                            [0] * len(self.tokenizer(token)['input_ids'][1:-1]))
                cap_labels = cap_labels + [0] # SEP
            cap_labels = torch.Tensor(cap_labels).unsqueeze(0)

        # MlM
        if self.args.do_nsp:
            encode_input = self.tokenizer.encode(text)
            encode_input_nsp = self.tokenizer.encode(next_sentence)

            token_type_ids = [0]*len(encode_input)
            token_type_ids = token_type_ids + [1]*(len(encode_input_nsp)-1)
            
            encode_input = encode_input + encode_input_nsp[1:]
            attention_mask = [1]*len(encode_input)

            inputs = torch.tensor(encode_input).unsqueeze(0)
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        else:
            encode_input = self.tokenizer(text, return_tensors='pt')

            inputs = encode_input['input_ids']
            attention_mask = encode_input['attention_mask']
            token_type_ids = encode_input['token_type_ids']

        if len(inputs[0]) > self.args.max_seq_len:
            inputs = inputs[0][:self.args.max_seq_len-1]
            inputs = torch.cat(
                (inputs, torch.tensor([self.tokenizer.sep_token_id])))

            token_type_ids = token_type_ids[0][:self.args.max_seq_len]
            attention_mask = attention_mask[0][:self.args.max_seq_len]

            inputs = inputs.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            if self.args.do_cap:
                cap_labels = cap_labels[0][:self.args.max_seq_len]

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # Padding
        padding_length = self.args.max_seq_len - len(inputs[0])
        if padding_length > 0:
            padding = torch.full((1, padding_length),
                                 self.tokenizer.pad_token_id, dtype=torch.long)
            inputs = torch.cat((inputs, padding), 1)
            token_type_ids = torch.cat((token_type_ids, padding), 1)
            attention_mask = torch.cat((attention_mask, padding), 1)

            padding_labels = torch.full(
                (1, padding_length), -100, dtype=torch.long)
            labels = torch.cat((labels, padding_labels), 1)
            if self.args.do_cap:
                cap_labels = torch.cat((cap_labels, torch.zeros(padding_labels.size())), 1)

        inputs = inputs.squeeze(0)
        token_type_ids = token_type_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        labels = labels.squeeze(0)

        if self.args.do_cap:
            cap_labels = cap_labels.squeeze(0)

        # print(labels.shape,cap_labels.shape,label_nsp.shape)
        if self.args.do_cap and self.args.do_nsp:
            return inputs, token_type_ids, attention_mask, labels, cap_labels, label_nsp
        elif self.args.do_cap:
            return inputs, token_type_ids, attention_mask, labels, cap_labels
        elif self.args.do_nsp:
            return inputs, token_type_ids, attention_mask, labels, label_nsp
        else:
            return inputs, token_type_ids, attention_mask, labels

    
