import logging
import os
import random

import numpy as np
import torch
from model import phoBERT, ViHnBERT
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer,
    RobertaConfig
)

MODEL_CLASSES = {
    "vihnbert": (RobertaConfig, ViHnBERT, AutoTokenizer),
    "phobert": (RobertaConfig, phoBERT, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "vihnbert": "demdecuong/vihealthbert-base-word",
    "phobert": "vinai/phobert-base"
}

def get_slot_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.slot_label_file), "r", encoding="utf-8")
    ]

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


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

def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    results.update(slot_result)
    return results

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    print(classification_report(labels, preds, digits=4))
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }
