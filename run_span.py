import os
import sys
import time
import json
import glob
import random
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np

# datasets
from transformers.data import DataProcessor

# models
import transformers
from transformers import WEIGHTS_NAME
from transformers import (
    BertConfig, 
    BertTokenizer,
    BertPreTrainedModel,
    BertModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

# trainer & training arguments
from transformers import AdamW, get_linear_schedule_with_warmup
from lamb import Lamb

# metrics
from seqeval.metrics.sequence_labeling import (
    accuracy_score,
    classification_report, 
    performance_measure,
    f1_score, precision_score, recall_score,
    get_entities
)
from utils import  LABEL_MEANING_MAP

class BertConfigSpanV2(BertConfig):

    def __init__(self, 
        max_span_length=10, 
        width_embedding_dim=150,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_span_length = max_span_length
        self.width_embedding_dim = width_embedding_dim

# from allennlp.nn.util import batched_index_select
def batched_index_select(input, index):
    batch_size, sequence_length, hidden_size = input.size()
    batch_size, num_spans = index.size()
    index_onehot = torch.FloatTensor(
        batch_size, num_spans, sequence_length).to(input.device)
    index_onehot.zero_()
    index_onehot.scatter_(2, index.unsqueeze(2), 1)
    output = torch.bmm(index_onehot, input)
    return output

class SpanV2(nn.Module):
    
    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_dim):
        super(SpanV2, self).__init__()

        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + width_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, hidden_states, spans):
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(hidden_states, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(hidden_states, spans_end)
        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)
        spans_embedding = torch.cat([
            spans_start_embedding, 
            spans_end_embedding, 
            spans_width_embedding
        ], dim=-1)  # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)
        return logits
        
    def decode_batch(self,
        batch,      # (batch_size, num_spans, num_labels)
        spans,      # (batch_size, num_spans, 3)
        span_mask,  # (batch_size, num_spans)
        is_logits:  bool=True,
    ):
        decodeds = []
        if is_logits:
            labels = batch.argmax(dim=-1)
        else:
            labels = batch
        for labels_, spans_, span_mask_ in zip(labels, spans, span_mask):
            span_mask_ = span_mask_ == 1.
            labels_ = labels_[span_mask_].cpu().numpy().tolist()
            spans_ = spans_[span_mask_].cpu().numpy().tolist()

            decoded_ = []
            for t, s in zip(labels_, spans_):
                decoded_.append([t, s[0] - 1, s[1] - 1])
            decodeds.append(decoded_)
        
        return decodeds

class SpanV2Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, 
        logits=None,        # (batch_size, num_spans, num_labels)
        label=None,         # (batch_size, num_spans)
        mask=None,          # (batch_size, num_spans)
    ):
        num_labels = logits.size(-1)
        loss_mask = mask.view(-1) == 1
        loss = self.loss_fct(logits.view(-1, num_labels), label.view(-1))
        loss = loss[loss_mask].mean()
        return loss


class BertSpanV2ForNer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span = SpanV2(config.hidden_size, config.num_labels, 
            config.max_span_length, config.width_embedding_dim)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        spans=None,         # (batch_size, num_spans, 3)
        span_mask=None,     # (batch_size, num_spans)
        label=None,         # (batch_size, num_spans)
        input_len=None,     # (batch_size)
        sent_start=None,    # (batch_size)
        sent_end=None,      # (batch_size)
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.span(sequence_output, spans)  # (batch_size, num_spans, num_labels)

        total_loss = None
        if label is not None:
            loss_fct = SpanV2Loss()
            total_loss = loss_fct(logits, label, span_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class NerArgumentParser(ArgumentParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_args_from_json(self, json_file):
        data = json.loads(Path(json_file).read_text())
        return Namespace(**data)
    
    def save_args_to_json(self, json_file, args):
        Path(json_file).write_text(json.dumps(vars(args), indent=4))

    def build_arguments(self):

        # Required parameters
        self.add_argument("--version", default=None, type=str, required=True,
                            help="Version of training model.")
        self.add_argument("--device", default=None, type=str, required=True,
                            help="Device for training.")
        self.add_argument("--n_gpu", default=1, type=int, required=True,
                            help="Device for training.")
        self.add_argument("--task_name", default=None, type=str, required=True,
                            help="The name of the task to train selected in the list: ")
        self.add_argument("--dataset_name", default=None, type=str, required=True,
                            help="The name of the dataset for the task")
        self.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
        self.add_argument("--train_file", default=None, type=str, required=True)
        self.add_argument("--dev_file", default=None, type=str, required=True)
        self.add_argument("--test_file", default=None, type=str, required=True)
        self.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list: ")
        self.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: " )
        self.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.", )
        
        self.add_argument("--max_span_length", default=10, type=int)
        self.add_argument("--width_embedding_dim", default=150, type=int)
        self.add_argument("--optimizer", default="adamw", type=str)
        
        # Other parameters
        self.add_argument('--scheme', default='IOB2', type=str,
                            choices=['IOB2', 'IOBES'])
        self.add_argument('--loss_type', default='ce', type=str,
                            choices=['lsr', 'focal', 'ce'])
        self.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        self.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name", )
        self.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3", )
        self.add_argument("--train_max_seq_length", default=128, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        self.add_argument("--eval_max_seq_length", default=512, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        self.add_argument("--do_train", action="store_true",
                            help="Whether to run training.")
        self.add_argument("--do_eval", action="store_true",
                            help="Whether to run eval on the dev set.")
        self.add_argument("--do_predict", action="store_true",
                            help="Whether to run predictions on the test set.")
        self.add_argument("--evaluate_during_training", action="store_true",
                            help="Whether to run evaluation during training at each logging step.", )
        self.add_argument("--evaluate_each_epoch", action="store_true",
                            help="Whether to run evaluation during training at each epoch, `--logging_step` will be ignored", )
        self.add_argument("--do_lower_case", action="store_true",
                            help="Set this flag if you are using an uncased model.")
                            
        # adversarial training
        self.add_argument("--do_fgm", action="store_true",
                            help="Whether to adversarial training.")
        self.add_argument('--fgm_epsilon', default=1.0, type=float,
                            help="Epsilon for adversarial.")
        self.add_argument('--fgm_name', default='word_embeddings', type=str,
                            help="name for adversarial layer.")

        self.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        self.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        self.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.", )
        self.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        self.add_argument("--other_learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for crf and linear layer.")
        self.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay if we apply some.")
        self.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        self.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        self.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        self.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

        self.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
        self.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
        self.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
        self.add_argument("--save_best_checkpoints", action="store_true", help="Save best checkpoint each `--logging_steps`, `--save_step` will be ignore")
        self.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
        self.add_argument("--predict_checkpoints", type=int, default=0,
                            help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
        self.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        self.add_argument("--overwrite_output_dir", action="store_true",
                            help="Overwrite the content of the output directory")
        self.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        self.add_argument("--fp16", action="store_true",
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
        self.add_argument("--fp16_opt_level", type=str, default="O1",
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html", )
        self.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

        return self

class NerProcessor(DataProcessor):

    def get_train_examples(self, data_dir, data_file):
        """Gets a collection of :class:`InputExample` for the train set."""
        return list(self._create_examples(data_dir, data_file, 'train'))

    def get_dev_examples(self, data_dir, data_file):
        """Gets a collection of :class:`InputExample` for the dev set."""
        return list(self._create_examples(data_dir, data_file, 'dev'))

    def get_test_examples(self, data_dir, data_file):
        """Gets a collection of :class:`InputExample` for the test set."""
        return list(self._create_examples(data_dir, data_file, 'test'))
    
    @property
    def label2id(self):
        return {label: i for i, label in enumerate(self.get_labels())}
    
    @property
    def id2label(self):
        return {i: label for i, label in enumerate(self.get_labels())}

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    def _create_examples(self, data_dir, data_file, mode):
        raise NotImplementedError()
    
    def entities2tags(self, entities, seq_len):
        ner_tags = ["O"] * seq_len
        for t, s, e in entities:
            if s < 0 or s >= seq_len or e < 0 or e >= seq_len \
                or s > e or ner_tags[s] != "O" or ner_tags[e] != "O":
                continue
            ner_tags[s] = f"B-{t}"
            for i in range(s + 1, e + 1):
                ner_tags[i] = f"I-{t}"
        return ner_tags

class CailNerProcessor(NerProcessor):

    def get_labels(self):
        return [
            "X", "O", "[START]", "[END]",
        ] + list(LABEL_MEANING_MAP.keys())
    
    def _create_examples(self, data_dir, data_file, mode):
        data_path = os.path.join(data_dir, data_file)
        with open(data_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f.readlines()]
            for sentence_counter, line in enumerate(lines):
                sentence = (
                    sentence_counter,
                    {
                        "id": f"{mode}-{str(line['id'])}",
                        "tokens": list(line["text"]),
                        "entities": line.get("entities", None) 
                            if mode in ["train", "dev"] else None,
                        "sent_start": line["sent_start"],
                        "sent_end": line["sent_end"],
                    }
                )
                yield sentence

class NerDataset(torch.utils.data.Dataset):

    def __init__(self, examples, process_pipline=[]):
        super().__init__()
        self.examples = examples
        self.process_pipline = process_pipline

    def __getitem__(self, index):
        # get example
        example = self.examples[index]
        # preprocessing
        for proc in self.process_pipline:
            example = proc(example)
        # convert to features
        return example
    
    def __len__(self):
         return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        max_len = max([b["input_len"] for b in batch])[0].item()
        collated = dict()
        for k in ["input_ids", "token_type_ids", "attention_mask", "input_len", "sent_start", "sent_end"]:
            t = torch.cat([b[k] for b in batch], dim=0)
            if k not in ["input_len", "sent_start", "sent_end"]:
                t = t[:, :max_len] # dynamic batch
            collated[k] = t
        for k in ["spans", "span_mask", "label"]:
            if batch[0][k] is None:
                collated[k] = None
                continue
            t = pad_sequence([b[k][0] for b in batch], batch_first=True)
            collated[k] = t
        return collated

class Example2Feature:
    
    def __init__(self, tokenizer, label2id, max_seq_length, max_span_length):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
    
    def __call__(self, example):
        return self._convert_example_to_feature(example)
    
    def _encode_span(self, max_length, input_len, sent_start, sent_end):
        spans = []; span_mask = []
        for i in range(sent_start, sent_end):
            for j in range(i, min(min(max_length, sent_end), i + self.max_span_length)):
                spans.append([i, j, j - i + 1])
                span_mask.append(0 if i >= input_len else 1)
        spans = torch.tensor([spans])               # (1, num_spans, 3) 
        span_mask = torch.tensor([span_mask])       # (1, num_spans)
        return spans, span_mask

    def _encode_label(self, entities, spans):
        tag_o = self.label2id["O"]
        entities = {(b + 1, e + 1): self.label2id[t] for t, b, e, _ in entities}
        label = [entities.get((b, e), tag_o) for b, e, l in spans[0]]
        label = torch.tensor([label])               # (1, num_spans)
        return label

    def _convert_example_to_feature(self, example):
        id_ = example[1]["id"]
        tokens = example[1]["tokens"]
        entities = example[1]["entities"]
        sent_start = example[1]["sent_start"]
        sent_end = example[1]["sent_end"]

        # encode input
        inputs = self.tokenizer.encode_plus(
            text=tokens,
            text_pair=None, 
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_seq_length,
            is_split_into_words=True,
            return_tensors="pt",
        )
        inputs["input_len"] = inputs["attention_mask"].sum(dim=1)  # for special tokens
        input_len = inputs["input_len"].item()
        inputs["spans"], inputs["span_mask"] = self._encode_span(
            input_len, input_len, sent_start, sent_end)  # dynamic batch
        inputs["sent_start"] = torch.tensor([sent_start])
        inputs["sent_end"] = torch.tensor([sent_end])
        
        if entities is None:
            inputs["label"] = None
            return inputs

        # encode label
        inputs["label"] = self._encode_label(entities, 
            inputs["spans"].cpu().numpy().tolist())
        return inputs

class FGM():
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def seed_everything(seed=None, reproducibility=True):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    '''
    if seed is None:
        seed = int(_select_seed_randomly())
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def init_logger(name, log_file='', log_file_level=logging.NOTSET):
    '''
    初始化logger
    '''
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                    datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger

def train(args, model, processor, tokenizer):
    """ Train the model """
    train_dataset = load_dataset(args, processor, tokenizer, data_type='train')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=NerDataset.collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.evaluate_each_epoch:
        args.logging_steps = args.save_steps = int(t_total // args.num_train_epochs)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    bert_param_optimizer_ids = [id(p) for n, p in bert_param_optimizer]
    other_param_optimizer = [(n, p) for n, p in model.named_parameters() if id(p) not in bert_param_optimizer_ids]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in other_param_optimizer], 
         'weight_decay': args.weight_decay, 'lr': args.other_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    if args.optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "lamb":
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.do_fgm:
        fgm = FGM(model, emb_name=args.fgm_name, epsilon=args.fgm_epsilon)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, best_f1 = 0.0, 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_no in range(int(args.num_train_epochs)):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training...')
        for step, batch in pbar:
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items() if v is not None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                if args.model_type.split('_')[0] in ["roberta", "xlnet"]:
                    batch["token_type_ids"] = None

            outputs = model(**batch)
            loss = outputs['loss']  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.do_fgm:
                fgm.attack()
                loss_adv = model(**batch)[0]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                if args.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss_adv, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_adv.backward()
                fgm.restore()
            pbar.set_description(desc=f"Training[{epoch_no}]... loss={loss.item():.6f}")
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.evaluate_during_training and \
                    args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        eval_results = evaluate(args, model, processor, tokenizer)
                        logger.info(f"loss={eval_results.pop('loss')}")
                        for entity, metrics in eval_results.items():
                            logger.info("{:*^50s}".format(entity))
                            logger.info("\t".join(f"{metric:s}={value:f}" 
                                for metric, value in metrics.items()))
                        if args.save_best_checkpoints:
                            if eval_results["micro avg"]["f1-score"] > best_f1:
                                best_f1 = eval_results["micro avg"]["f1-score"]
                                # Save model checkpoint
                                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(999999))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.json"))
                                logger.info("Saving model checkpoint to %s", output_dir)
                                tokenizer.save_vocabulary(output_dir)
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                if args.local_rank in [-1, 0] and not args.save_best_checkpoints and \
                    args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.json"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(args, model, processor, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_dataset(args, processor, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=NerDataset.collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Eval...')
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    y_true = []; y_pred = []
    id2label = processor.id2label
    for step, batch in pbar:
        # forward step
        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items() if v is not None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                if args.model_type.split('_')[0] in ["roberta", "xlnet"]:
                    batch["token_type_ids"] = None
            outputs = model(**batch)
            tmp_eval_loss = outputs['loss']
            logits = outputs['logits']
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        # calculate metrics
        preds = model.span.decode_batch(logits, batch["spans"], batch["span_mask"])
        for pred_no, (pred, input_len, start, end) in enumerate(zip(
                preds, batch["input_len"], batch["sent_start"], batch["sent_end"])):
            pred = [(LABEL_MEANING_MAP[id2label[t]], b, e) for t, b, e in pred if id2label[t] != "O"]
            pred = processor.entities2tags(pred, input_len - 2)
            y_pred.append(pred[start: end])

        labels = model.span.decode_batch(batch["label"], batch["spans"], batch["span_mask"], 
            batch["sent_start"], batch["sent_end"], is_logits=False)
        for label_no, (label, input_len, start, end) in enumerate(zip(
                labels, batch["input_len"], batch["sent_start"], batch["sent_end"])):
            label = [(LABEL_MEANING_MAP[id2label[t]], b, e) for t, b, e in label if id2label[t] != "O"]
            label = processor.entities2tags(label, input_len - 2)
            y_true.append(label[start: end])

    results = classification_report(y_true, y_pred, digits=6, output_dict=True, scheme=args.scheme)
    results['loss'] = eval_loss / nb_eval_steps
    return results

def predict(args, model, processor, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_dataset(args, processor, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=NerDataset.collate_fn)
    id2label = processor.id2label
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")

    if isinstance(model, nn.DataParallel):
        model = model.module
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Predicting...")
    for step, batch in pbar:
        model.eval()
        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items() if v is not None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                if args.model_type.split('_')[0] in ["roberta", "xlnet"]:
                    batch["token_type_ids"] = None
            outputs = model(**batch)
            logits = outputs['logits']

        preds = model.span.decode_batch(logits, batch["spans"], batch["span_mask"])
        pred, input_len = preds[0], batch["input_len"][0]
        pred = [(id2label[t], b, e) for t, b, e in pred if id2label[t] != "O"]
        pred = processor.entities2tags(pred, input_len - 2), 
        pred = pred[batch["sent_start"][0]: batch["sent_end"][0]]
        label_entities_map = {label: [] for label in LABEL_MEANING_MAP.keys()}
        for t, b, e in get_entities(pred):
            label_entities_map[t].append(f"{b};{e}")
        entities = [{"label": label, "span": label_entities_map[label]} for label in LABEL_MEANING_MAP.keys()]
        # 预测结果文件为一个json格式的文件，包含两个字段，分别为``id``和``entities``
        results.append({
            "id": test_dataset.examples[step][1]["id"].split("-")[1],
            "entities": entities,
        })
    
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

PROCESSER_CLASS = {
    "cail_ner": CailNerProcessor,
}

MODEL_CLASSES = {
    "bert_span": (BertConfigSpanV2, BertSpanV2ForNer, BertTokenizer),
}

def load_dataset(args, processor, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir, args.train_file)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir, args.dev_file)
    else:
        examples = processor.get_test_examples(args.data_dir, args.test_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    max_seq_length = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    return NerDataset(examples, process_pipline=[
        Example2Feature(tokenizer, processor.label2id, max_seq_length, config.max_span_length),
    ])


if __name__ == "__main__":

    parser = NerArgumentParser()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_args_from_json(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.build_arguments().parse_args()
    # args = parser.parse_args_from_json(json_file="args/bert_span-baseline.json")

    # Set seed before initializing model.
    seed_everything(args.seed)
    
    # User-defined post initialization
    output_dir = f"{args.task_name}-{args.dataset_name}-{args.model_type}-{args.version}-{args.seed}"
    if not args.output_dir.endswith(output_dir):
        args.output_dir = os.path.join(args.output_dir, output_dir)
    args.logging_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    parser.save_args_to_json(os.path.join(args.output_dir, "training_args.json"), args)

    # Setup logging
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger = init_logger(__name__, log_file=os.path.join(args.output_dir, f'{time_}.log'))
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1
    # args.device = device
    args.device, args.n_gpu = torch.device(args.device), 1

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )

    # Prepare NER task
    args.dataset_name = args.dataset_name.lower()
    if args.dataset_name not in PROCESSER_CLASS:
        raise ValueError("Task not found: %s" % (args.dataset_name))
    processor = PROCESSER_CLASS[args.dataset_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, max_span_length=args.max_span_length,
                                          cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, processor, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, processor, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, processor, tokenizer, prefix=prefix)


