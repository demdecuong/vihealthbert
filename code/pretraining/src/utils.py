import logging

from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from model import ViHnBERT

MODEL_CLASSES = {
    "phobert": (RobertaConfig, ViHnBERT, AutoTokenizer),
    "hnbert": (RobertaConfig, ViHnBERT, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "phobert" : "vinai/phobert-base",
    "hnbert": "demdecuong/vihealthbert-base-word"
}

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def read_line_by_line(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(' '.join(line.split()).strip().replace('\n',''))
    return data