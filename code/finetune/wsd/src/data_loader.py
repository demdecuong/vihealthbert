from torch.utils.data import Dataset
import torch

import random
import os
import json
import copy
import re

import logging

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, id, text, text_tokens, expansion, start_char_idx, length_acronym, start_token_idx, end_token_idx, label) -> None:
        super().__init__()
        self.guid = guid
        self.id = id
        self.text = text
        self.text_tokens = text_tokens
        self.expansion = expansion
        self.start_char_idx = start_char_idx
        self.length_acronym = length_acronym
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, id, input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx, label, expansion):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.label = label
        self.expansion = expansion

        self.id = id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):
    """Processor for the ArcBERT data set """
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    @classmethod
    def _read_file(cls, input_file):
        """Reads json file"""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    
    def clean_syn(self, text):
            text = re.sub('[\?,\.\!:;\(\)]', '', text)
            return text

    def _create_examples(self, data, mode):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, example in enumerate(data):
            guid = "%s-%s" % (mode, i)
            id = example['id']
            # 1. Input text
            text = example['text']
            text_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in text:
                if self.is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        text_tokens.append(c)
                    else:
                        text_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(text_tokens)-1)
            # 2. Expansion of acr
            expansion = example['expansion']
            # 3. Position of acr and acr
            start_char_idx = example['start_char_idx']
            length_acronym = example['length_acronym']
            start_token_idx = char_to_word_offset[start_char_idx]
            end_token_idx = char_to_word_offset[start_char_idx+length_acronym-1]
            # 4. Label
            label = example['label']
            examples.append(InputExample(
                guid=guid,
                id=id,
                text=text,
                text_tokens=text_tokens,
                expansion=expansion,
                start_char_idx=start_char_idx,
                length_acronym=length_acronym,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                label=label
            ))
        return examples
    
    def get_examples(self, args, mode):
        """
        Args:
            mode: train, dev, test
        """

        data_path = os.path.join(args.data_dir, mode)
        data = self._read_file(os.path.join(data_path, args.data_file_name))
        
        PATH_DICTIONARY = os.path.join(args.data_dir, args.dict_file_name)
        if not os.path.isfile(PATH_DICTIONARY):
            raise Exception(f"Folder {args.data_dir} doesn't contain canonical dictionary")
        dictionary = self._read_file(PATH_DICTIONARY)
        
        examples = []

        pos_data = add_label_positive_sample(data)
        examples.extend(pos_data)

        neg_data = negative_data(pos_data, dictionary, mode)
        examples.extend(neg_data)
            
        return self._create_examples(examples, mode)



def negative_data(positive_data:list, diction:dict, mode) -> list:
    """
    Funciton: Create negative samples
    args:
        positive_data: training data whose format {
            'acronym': ...,(optional)
            'expansion': ...,
            'text': ...,
            'start_char_idx: ...,
            'lenght_acronym': ...,
            'label': 1 (positive sample)
        }
        and
        diction: dictionary of acronym and able expansion respectively
    """

    neg_data = []
    tmp = 0
    for sample in positive_data:
        try:
            acronym = sample["text"][sample["start_char_idx"]:sample["start_char_idx"]+sample['length_acronym']]
            list_neg_expansion = diction[acronym].copy()
            list_neg_expansion.remove(sample["expansion"])
            if mode == 'train':
                if len(list_neg_expansion) > 1: 
                    list_neg_expansion = random.sample(list_neg_expansion, random.randint(1,2))
            elif mode == 'dev' or mode == 'test':
                if len(list_neg_expansion) > 1:
                    list_neg_expansion = list_neg_expansion
            for i in list_neg_expansion:
                neg_data.append(sample.copy())
                neg_data[tmp]["expansion"] = i
                neg_data[tmp]["label"] = 0 # pseudo negative samples
                tmp += 1
        except: 
            print(sample)
            continue
    
    return neg_data

def add_label_positive_sample(data: list):
    for idx, sample in enumerate(data):
        sample['text'] = sample['text'].lower()
        sample['label'] = 1
        sample['id'] = idx
    return data



def convert_examples_to_features(examples,
                                max_seq_len,
                                tokenizer):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        orig_to_tok_index = []
        all_doc_tokens = []

        for (i, token) in enumerate(example.text_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)
        
        start_token_idx = orig_to_tok_index[example.start_token_idx]
        if len(orig_to_tok_index) == (example.end_token_idx + 1):
            end_token_idx = orig_to_tok_index[-1]
        else:
            end_token_idx = orig_to_tok_index[example.end_token_idx + 1] - 1
        
        input_ids = []
        
        input_ids += [cls_token]
        input_ids += all_doc_tokens
        input_ids += [sep_token]
        
        token_type_ids = [0]*len(input_ids)
        
        expansion = example.expansion
        expansion_tokens = tokenizer.tokenize(expansion)
        
        input_ids += expansion_tokens
        input_ids += [sep_token]
        

        token_type_ids += [1]*(len(expansion_tokens) + 1)

        attention_mask = [1]*len(input_ids)
        
        input_ids = tokenizer.convert_tokens_to_ids(input_ids)
        
        padding = max_seq_len - len(input_ids)
        
        if padding < 0:
            print('Ignore sample has length > 256 tokens')
            continue
        
        input_ids = input_ids + ([pad_token_id] * padding)
        attention_mask = attention_mask + [0]*padding
        token_type_ids = token_type_ids + [0]*padding
        assert len(input_ids) == len(attention_mask) == len(token_type_ids), "Error with input length {} vs attention mask length {}, token type length {}".format(len(input_ids), len(attention_mask), len(token_type_ids))
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        id = example.id
        label=example.label
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in all_doc_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("labels: %s" % " ".join([str(x) for x in [label]]))
            logger.info("expansion: %s" % " ".join([str(x) for x in [expansion]]))

        
        features.append(
                InputFeatures(
                    id = id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    label=label,
                    expansion=expansion
                )
            )
    return features

def load_and_cache_examples(args, tokenizer, mode=None):
    if not mode:
        return None
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(mode,
                                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                args.max_seq_len
                                )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(args, "train")
        elif mode == "dev":
            examples = processor.get_examples(args, "dev")
        elif mode == "test":
            examples = processor.get_examples(args, "test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
            
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer
        )

        logger.info("Saving features into cached file %s", cached_features_file)

        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.int64)
    all_start_token_idx = torch.tensor([f.start_token_idx for f in features], dtype=torch.int64)
    all_end_token_idx = torch.tensor([f.end_token_idx for f in features], dtype=torch.int64)
    all_label = torch.tensor([f.label for f in features], dtype=torch.float)

    all_id = torch.tensor([f.id for f in features], dtype=torch.long)
    all_expansion = [f.expansion for f in features]

    return (
        all_input_ids,
        all_token_type_ids,
        all_attention_mask,
        all_start_token_idx,
        all_end_token_idx,
        all_label,
        all_id,
        all_expansion
    )


class AcrDataset(Dataset):
    def __init__(self,
                args,
                tokenizer,
                mode) -> None:
        super().__init__()
        self.mode = mode
        
        self.dataset = load_and_cache_examples(args, tokenizer, mode)

    def __len__(self) -> int:
        return len(self.dataset[0])
    
    def __getitem__(self, index: int):
        return  self.dataset[0][index], self.dataset[1][index], self.dataset[2][index], self.dataset[3][index], self.dataset[4][index], self.dataset[5][index], self.dataset[6][index], self.dataset[7][index]

        