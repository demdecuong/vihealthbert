from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch

from early_stopping import EarlyStopping
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
from transformers.trainer_utils import set_seed
from transformers.trainer_pt_utils import get_parameter_names
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.file_utils import WEIGHTS_NAME

from utils import MODEL_CLASSES

import logging
import os
from tqdm.auto import tqdm, trange
from typing import Optional

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    make the pretrained Health News Bert in VietNam domain with LM training methods:

        1. Words aware:
            - token masked language model

    """

    def __init__(self,
                 args,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None) -> None:
        super().__init__()

        self.args = args
        set_seed(self.args.seed)

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model_type]

        if args.pretrained:
            print(args.model_name_or_path)
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                args=args
            )
        else:
            self.config = self.config_class.from_pretrained(
                args.model_name_or_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args
            )

        # GPU or CPU
        torch.cuda.set_device(self.args.gpu_id)
        print('GPU ID :', self.args.gpu_id)
        print('Cuda device:', torch.cuda.current_device())
        self.device = args.device

        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # writer = SummaryWriter(log_dir=self.args.model_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_loader) //
                                        self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(
                train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        optimizer = self.create_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d",
                    self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(
            patience=self.args.early_stopping, verbose=True)

        tr_loss = 0.0
        tr_mlm_loss = 0.0
        tr_cap_loss = 0.0
        tr_nsp_loss = 0.0

        global_step = 0.0

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_loader, desc="Iteration", position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "token_type_ids": batch[1].to(self.device),
                    "attention_mask": batch[2].to(self.device),
                    "labels_mlm": batch[3].to(self.device),
                }
                if self.args.do_cap and self.args.do_nsp:
                    inputs["labels_cap"] = batch[4].to(self.device)
                    inputs["labels_nsp"] = batch[5].to(self.device)
                elif self.args.do_cap:
                    inputs["labels_cap"] = batch[4].to(self.device)
                elif self.args.do_nsp:
                    inputs["labels_nsp"] = batch[4].to(self.device)


                outputs = self.model(**inputs)

                # Mask language model loss
                total_loss = outputs[0]

                mlm_loss = outputs[1]
                tr_mlm_loss += mlm_loss.item()

                if self.args.do_cap and self.args.do_nsp:
                    cap_loss = outputs[2]
                    tr_cap_loss += cap_loss.item()
                    nsp_loss = outputs[3]
                    tr_nsp_loss += nsp_loss.item()
                elif self.args.do_cap:
                    cap_loss = outputs[2]
                    tr_cap_loss += cap_loss.item()          
                elif self.args.do_nsp:
                    nsp_loss = outputs[2]
                    tr_nsp_loss += nsp_loss.item()

                # Back - propagation
                if self.args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.args.gradient_accumulation_steps

                total_loss.backward()
                tr_loss += total_loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient cliping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    self.model.zero_grad()
                    global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    dev_result = self.evaluate('test')
                    # pass
                    print(f'[TRAIN] Loss : {tr_loss/global_step}')
                    if self.args.do_cap or self.args.do_nsp:
                        print(f'[TRAIN] Loss MLM : {mlm_loss/global_step}')

                    if self.args.do_cap:
                        print(f'[TRAIN] Loss Capitalized Prediction: {cap_loss/global_step}')

                    if self.args.do_nsp:
                        print(f'[TRAIN] Loss NSP : {tr_nsp_loss/global_step}')

                    early_stopping(dev_result, self.model.roberta, self.args)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        # exit()
                        break
                # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        print(tr_loss/global_step)
        return global_step, tr_loss

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_loader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()

        eval_loss = 0.0
        eval_mlm_loss = 0.0
        eval_cap_loss = 0.0
        eval_nsp_loss = 0.0

        nb_eval_steps = 0

        for batch in tqdm(eval_loader):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "token_type_ids": batch[1].to(self.device),
                    "attention_mask": batch[2].to(self.device),
                    "labels_mlm": batch[3].to(self.device),
                }
                if self.args.do_cap and self.args.do_nsp:
                    inputs["labels_cap"] = batch[4].to(self.device)
                    inputs["labels_nsp"] = batch[5].to(self.device)
                elif self.args.do_cap:
                    inputs["labels_cap"] = batch[4].to(self.device)
                elif self.args.do_nsp:
                    inputs["labels_nsp"] = batch[4].to(self.device)

                outputs = self.model(**inputs)

                total_loss =  outputs[0]
                eval_loss += total_loss.item()

                # Mask language model loss
                mlm_loss = outputs[1]
                eval_mlm_loss += mlm_loss.item()

                if self.args.do_cap and self.args.do_nsp:
                    cap_loss = outputs[2]
                    eval_cap_loss += cap_loss.item()
                    nsp_loss = outputs[3]
                    eval_nsp_loss += nsp_loss.item()
                elif self.args.do_cap:
                    cap_loss = outputs[2]
                    eval_cap_loss += cap_loss.item()
                if self.args.do_nsp:
                    nsp_loss = outputs[2]
                    eval_nsp_loss += nsp_loss.item()

            nb_eval_steps += 1
        
        print(f'[EVALUATE] Loss : {eval_loss/nb_eval_steps}')

        if self.args.do_cap or self.args.do_nsp:
            print(f'[EVALUATE] Loss MLM : {eval_mlm_loss/nb_eval_steps}')

        if self.args.do_cap:
            print(f'[EVALUATE] Loss Capitalized Prediction: {eval_cap_loss/nb_eval_steps}')
      
        if self.args.do_nsp:
            print(f'[EVALUATE] Loss NSP : {nsp_loss/nb_eval_steps}')

        return eval_loss/nb_eval_steps

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info(f"Saving model checkpoint to {output_dir}")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def create_optimizer(self):
        # decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = list(self.model.named_parameters())
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        optimizer_kwargs["lr"] = self.args.learning_rate
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):

        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer)
