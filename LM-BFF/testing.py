from transformers import set_seed
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, XLMRobertaForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from sklearn.metrics import classification_report

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        # Note: the eval dataloader is sequential, so the examples are in order.
        # We average the logits over each sample for using demonstrations.
        predictions = p.predictions
        num_logits = predictions.shape[-1]
        logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
        logits = logits.mean(axis=0)
        
        if num_logits == 1:
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)

        # Just for sanity, assert label ids are the same.
        label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
        label_ids_avg = label_ids.mean(axis=0)
        label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
        assert (label_ids_avg - label_ids[0]).mean() < 1e-2
        label_ids = label_ids[0]

        return compute_metrics_mapping[task_name](task_name, preds, label_ids)

    return compute_metrics_fn

def build_compute_metrics_report_fn(task_name: str, output_dir) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        # Note: the eval dataloader is sequential, so the examples are in order.
        # We average the logits over each sample for using demonstrations.
        predictions = p.predictions
        num_logits = predictions.shape[-1]
        logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
        logits = logits.mean(axis=0)
        
        if num_logits == 1:
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)

        # Just for sanity, assert label ids are the same.
        label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
        label_ids_avg = label_ids.mean(axis=0)
        label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
        assert (label_ids_avg - label_ids[0]).mean() < 1e-2
        label_ids = label_ids[0]
        report = classification_report(label_ids, preds)
        print(report)
        output_test_file = os.path.join(
            training_args.output_dir, f"test_results_{test_dataset.args.task_name}.classification_report.txt"
        )
        with open(output_test_file, 'w') as writer:
            writer.write(report)

        return compute_metrics_mapping[task_name](task_name, preds, label_ids)
    return compute_metrics_fn

prompt_path = 'experiments/english_xlmr/prompt/toxic/16-13.top.txt'
with open(prompt_path) as file:
    for line in file:
        line = line.strip()
        template, mapping = line.split('\t')
print(template, mapping)
set_seed(13)
num_labels = 2
output_mode = 'classification'
config = AutoConfig.from_pretrained(
        'result/english-xlmr/toxic-prompt-16-13-roberta-base-8019/config.json',
        num_labels=num_labels,
        finetuning_task='toxic',
        cache_dir=None
    )
model_fn = RobertaForPromptFinetuning
special_tokens = []
tokenizer = AutoTokenizer.from_pretrained(
        'result/english-xlmr/toxic-prompt-16-13-roberta-base-8019/',
        additional_special_tokens=special_tokens,
        cache_dir=None,
    )

model = model_fn.from_pretrained(
        'result/english-xlmr/toxic-prompt-16-13-roberta-base-8019/',
        from_tf=bool(".ckpt" in 'result/english-xlmr/toxic-prompt-16-13-roberta-base-8019/'),
        config=config,
        cache_dir=None,
    )
model.label_word_list = [0, 1]

test_results = {}
