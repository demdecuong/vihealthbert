import logging
import random
from collections import defaultdict

import numpy as np
import torch
import json
import os

from transformers import (
    AutoTokenizer,
    RobertaConfig
)
from model import ViHnBERT


MODEL_CLASSES = {
    "vihnbert": (RobertaConfig, ViHnBERT, AutoTokenizer),
    "phobert": (RobertaConfig, ViHnBERT, AutoTokenizer)
    }

MODEL_PATH_MAP = {
    "vihnbert": "demdecuong/vihealthbert-base-word",
    "phobert": "vinai/phobert-base"
    }

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def score_expansion(gold, prediction):
    """
    gold, prediction is list
    """
    correct = 0
    for i in range(len(gold)):
        if gold[i] == prediction[i]:
            correct += 1
    acc = correct/len(prediction)

    expansions = set()
    correct_per_expansion = defaultdict(int)
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)

    for i in range(len(gold)):
        expansions.add(gold[i])
        total_per_expansion[gold[i]] += 1
        pred_per_expansion[prediction[i]] += 1
        if gold[i] == prediction[i]:
            correct_per_expansion[gold[i]] += 1
    
    precs = defaultdict(int)
    recalls = defaultdict(int)

    for exp in expansions:
        precs[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    # micro-pred = micro-recall = micro-f1 = acc if len(gold) = len(prediction)
    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(total_per_expansion.values())
    micro_f1 = 2*micro_prec*micro_recall/(micro_prec+micro_recall) if micro_prec+micro_recall != 0 else 0

    # official evaluation metrics are the macro-averaged precision, recall and F1 for correct expansion predictions
    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = 2*macro_prec*macro_recall / (macro_prec+macro_recall) if macro_prec+macro_recall != 0 else 0

    return macro_prec, macro_recall, macro_f1, acc

def compute_metrics(args, ids, pred_expansions, pred_scores):
    thresh = args.threshold
    gold = read_json(os.path.join(args.data_dir, args.gold_file_name))
    pred = {}
    for i, expan, score in zip(ids, pred_expansions, pred_scores):
        if score > thresh:
            if i not in pred:
                pred[i] = [score, expan]
            else:
                if score > pred[i][0]:
                    pred[i] = [score, expan]
    pred = [pred[int(k)][1] if int(k) in pred else '' for k,v in gold.items()]
    gold = [gold[k] for k,v in gold.items()]
    assert len(gold) == len(pred)
    macro_prec, macro_recall, macro_f1, acc = score_expansion(gold, pred)
    result = {}

    result['macro_prec'] = macro_prec
    result['macro_recall'] = macro_recall
    result['macro_f1'] = macro_f1
    result['accuracy'] = acc

    return result
    
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
