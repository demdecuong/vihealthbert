import argparse
import torch

from data_loader import AcrDataset
from trainer import Trainer
from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, set_seed, load_tokenizer, compute_metrics
import logging

logger = logging.getLogger(__name__)

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = AcrDataset(args, tokenizer, 'train')
    dev_dataset = AcrDataset(args, tokenizer, 'dev')
    test_dataset = AcrDataset(args, tokenizer, 'test')

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    eval_loss, ids, pred_expansions, pred_scores = trainer.evaluate('test')
    results = compute_metrics(args, ids, pred_expansions, pred_scores)
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))


    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        eval_loss, ids, pred_expansions, pred_scores = trainer.evaluate('dev')
        results = compute_metrics(args, ids, pred_expansions, pred_scores)
        results['loss'] = eval_loss
        logger.info("***** Dev results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        eval_loss, ids, pred_expansions, pred_scores = trainer.evaluate('test')
        results = compute_metrics(args, ids, pred_expansions, pred_scores)
        results['loss'] = eval_loss
        logger.info("***** Test results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="../data", type=str, help="The input data dir")
    parser.add_argument("--data_file_name", default="data.json", type=str, help="The input data name")
    parser.add_argument("--gold_file_name", default="gold.json", type=str, help="The gold file name")
    parser.add_argument("--dict_file_name", default="dictionary.json", type=str, help="The dictionary file name")


    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--tuning_metric", default="macro_f1", type=str, help="Metrics to tune when training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=100.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=25,
        help="Number of unincreased validation step to wait for early stopping",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")
        
    # init pretrained
    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./workspace/vinbrain/vutran/Transfer_Learning/Domain_Adaptive/Finetune/WSD/src/XLMr_ADvn/1e-5/42/", type=str, help="The pretrained model path")

    parser.add_argument(
        "--threshold", default=0.5, type=float, help="Threshold"
    )

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    main(args)
