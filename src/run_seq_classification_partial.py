import datasets
from datasets import load_dataset

import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification

import numpy as np

from transformers import TrainingArguments, Trainer, HfArgumentParser, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from transformers import EvalPrediction, WEIGHTS_NAME
import torch
import torch.nn as nn 

import argparse
import os
import sys
import logging

from dataclasses import dataclass, field
from typing import Optional
import random
import glob
import json
from reindent import run as run_reindent
import io
import pdb
import scipy

from generate_partial import create_extended_dataset, create_grouped_indices

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    num_partials: int = field(
        default=3,
        metadata={
            "help": "The number of partials each code will generate. 3 means we take 0.25, 0.50, and 0.75"
            "of each code snippet, and the dataset will quadruple in size."
        },
    ) 
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    truncate_texta_from_first: bool = field(
        default=False,
        metadata={
            "help": "Truncate texta from the first when truncating. "
            "If False, will truncate texta from the last"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    labels_file: Optional[str] = field(default=None, metadata={"help": "A txt file containing the labels."})
    weights_file: Optional[str] = field(default=None, metadata={"help": "A txt file containing the weights."})
    grouped_indices_file: Optional[str] = field(default=None, metadata={"help": "A txt file containing the grouped indices."})
    grouped_labels_file: Optional[str] = field(default=None, metadata={"help": "A txt file containing the grouped labels."})
    predict_suffix: Optional[str] = field(default="", metadata={"help": "Suffix for predict file."})
    sentence1_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "Name of the key for sentence1 in the dataset" },
    )
    sentence2_key: Optional[str] = field(
        default="completion",
        metadata={"help": "Name of the key for sentence2 in the dataset" },
    )
    label_key: Optional[str] = field(
        default="binary_label",
        metadata={"help": "Name of the key for label in the dataset" },
    )

    def __post_init__(self):
        if (self.train_file is None and self.validation_file is None) and self.test_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def compute_metrics(p: EvalPrediction, compute_ranker_accuracy=False, grouped_indices=None, grouped_labels=None, pass_idx=1):
    # grouped_indices is a two-dimensional array where each row represents the indices 
    # of various datapoints in p that have the same prompt 
    
    # grouped_labels is the actual ternary labels of <prompt, completion> 
    # datapoints at the indices provided by grouped_indices

    pred_raw, labels = p    
    if compute_ranker_accuracy:
        # we will also compute the actual accuracy of the ranker
        pred_softmax = scipy.special.softmax(pred_raw, axis=1)
        pred_prob = pred_softmax[:,pass_idx] # prob of <prompt,completion> is Correct as predicted by the ranker
        prob_fun = lambda x: pred_prob[x] if x >=0 else 0 
        prob_fun = np.vectorize(prob_fun)
        grouped_prob = prob_fun(grouped_indices) # group prob predicted by task

        # top-1 accuracy
        best = np.argmax(grouped_prob, axis=1) # index of the best completion for each task as predicted by the ranker
        top1_label = grouped_labels[np.arange(len(grouped_labels)), best] # get the actual label of this best completion
        top1_accuracy = np.mean(top1_label == "Correct")
        top1_execution = 1.0 - np.mean(top1_label == "Execution error")

        
    pred = np.argmax(pred_raw, axis=1)
    pred_binary = pred == pass_idx
    labels_binary = labels == pass_idx
    accuracy = accuracy_score(y_true=labels_binary, y_pred=pred_binary)
    recall = recall_score(y_true=labels_binary, y_pred=pred_binary, average='micro')
    precision = precision_score(y_true=labels_binary, y_pred=pred_binary, average='micro')
    f1 = f1_score(y_true=labels_binary, y_pred=pred_binary, average='micro')

    # multi class predictions
    accuracy_mc = accuracy_score(y_true=labels, y_pred=pred)
    recall_mc = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision_mc = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1_mc = f1_score(y_true=labels, y_pred=pred, average='micro')

    metrics = {'f1': f1,
               'precision': precision,
               'recall': recall,
               'accuracy': accuracy,
               'f1_mc': f1_mc,
               'precision_mc': precision_mc,
               'recall_mc': recall_mc,
               'accuracy_mc': accuracy_mc,}

    if compute_ranker_accuracy:
        metrics['top1_accuracy'] = top1_accuracy
        metrics['top1_execution'] = top1_execution
    return metrics 

def reindent_code(codestr, replace_set=[]):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    out = ret.getvalue()
    for s in replace_set:
        out = out.replace(s, "")

    return out

# A trainer for balancing dataset with class weights
class CustomTrainer(Trainer):
    def set_weights(self, class_weights):
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    print("Helllllllooooooooooooo")
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
   
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    #transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can provide your own CSV/JSON training and evaluation files (see below)
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {}
    if training_args.do_train and data_args.train_file != None:
        data_files["train"] = data_args.train_file
    if training_args.do_eval and data_args.validation_file != None:
        data_files["validation"] = data_args.validation_file
    if training_args.do_predict and data_args.test_file != None:
        data_files["test"] = data_args.test_file


    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    print("Creating extended dataset")
    raw_datasets = create_extended_dataset(raw_datasets, n=data_args.num_partials) # new 

    # Labels
    if data_args.labels_file != None:
        # read labels from file
        with open(data_args.labels_file, 'r') as f:
            label_list = [line.strip() for line in f]
            num_labels = len(label_list)
    else:
        num_labels = 2
        label_list = [False, True]

    # grouped indices/labels for measuring actual accuracy of ranking models 
    if data_args.grouped_indices_file != None:
        orig_grouped_indices = np.load(data_args.grouped_indices_file)
        orig_grouped_labels = np.load(data_args.grouped_labels_file)
        print("Making new grouped indices and labels")
        grouped_indices, grouped_labels = create_grouped_indices(orig_grouped_indices, \
                                            orig_grouped_labels, 
                                            n=data_args.num_partials)
    else:
        grouped_indices = None
        grouped_labels = None
    # get the index of "Correct"  in the labels list
    pass_idx = label_list.index("Correct")

    print("label_list:", label_list)
    print("pass_idx:", pass_idx)


    if data_args.weights_file != None:
        # read weights from file
        with open(data_args.weights_file, 'r') as f:
            class_weights = [float(line.strip()) for line in f]
    else:
        class_weights = None 
    
    # Load pretrained model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,)    
    model = RobertaForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=num_labels, cache_dir=model_args.cache_dir)

    # import pdb
    # pdb.set_trace()

    # Preprocessing the raw datasets
    sentence1_key = data_args.sentence1_key #"prompt"
    sentence2_key = data_args.sentence2_key #"completion"
    label_key = data_args.label_key #"binary_label"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    label_to_id = {v: i for i, v in enumerate(label_list)} 
    model.config.label2id = label_to_id

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    mask_padding_with_zero = True
    pad_token = 0
    pad_token_segment_id = 0
    pad_on_left = False
    def preprocess_function(examples):

        # Tokenize the texts
        text1 = [reindent_code(s, replace_set=[]) for s in examples[sentence1_key]]
        text2 = [reindent_code(s, replace_set=[]) for s in examples[sentence2_key]]

        def trunc(tokens_a, tokens_b, max_length, truncate_texta_from_first=False):
            """Truncates a sequence pair in place to the maximum length."""
            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    if truncate_texta_from_first:
                        tokens_a.pop(0)
                    else:
                        tokens_a.pop()
                else:
                    tokens_b.pop()
        def custom_tokenize(text1, text2):
            all_input_ids = []
            all_attention_mask = []
            all_token_type_ids = []
            for i in range(len(text1)):
                tok_seq1 = tokenizer.tokenize(text1[i])
                tok_seq2 = tokenizer.tokenize(text2[i])
                
                trunc(tok_seq1, tok_seq2, max_seq_length - 3, truncate_texta_from_first=data_args.truncate_texta_from_first) # 3 is number of special tokens for bert sequence pair

                input_ids = [tokenizer.cls_token_id] 
                input_ids += tokenizer.convert_tokens_to_ids(tok_seq1)
                input_ids += [tokenizer.sep_token_id]

                token_type_ids = [0]*len(input_ids)

                input_ids += tokenizer.convert_tokens_to_ids(tok_seq2)
                input_ids += [tokenizer.sep_token_id]
                token_type_ids += [1]*(len(tok_seq2)+1) 

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([tokenizer.pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_token_type_ids.append(token_type_ids)

            result = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}
            return result

        result = custom_tokenize(text1, text2)
        
        # Map labels to IDs (not necessary for GLUE wc -s)
        if label_to_id is not None and label_key in examples:
            result["label"] = [(label_to_id[l]) for l in examples[label_key]]
        return result
        

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
            num_proc = 20,
        )
    if training_args.do_eval:
        if "validation" not in raw_datasets :
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on eval dataset",
            num_proc = 20,
        )
            

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            num_proc = 20,
        )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            input_ids = train_dataset[index]["input_ids"]
            text = tokenizer.decode(input_ids)
            print(text)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    if class_weights == None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=lambda x: compute_metrics(x, True, grouped_indices, grouped_labels, pass_idx),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=lambda x: compute_metrics(x, True, grouped_indices, grouped_labels, pass_idx),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.set_weights(class_weights)

    # Training
    if training_args.do_train:
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if training_args.do_train and last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        train_result = trainer.train(resume_from_checkpoint = last_checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # load model for eval and test
    # if output dir has a model, then load it
    if os.path.exists(os.path.join(training_args.output_dir, "pytorch_model.bin")):
        logger.info(f"Loading model from {os.path.join(training_args.output_dir, 'pytorch_model.bin')}")
        model = RobertaForSequenceClassification.from_pretrained(training_args.output_dir, num_labels=num_labels)
    else:
        # if last checkpoint exists and the output dir does not have a model, 
        # then we can load the best model using the trainer state in last checkpoint
        # Detecting last checkpoint.
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            with open(os.path.join(last_checkpoint, "trainer_state.json"), "r") as f:
                trainer_state = json.load(f)
                if "best_model_checkpoint" in trainer_state:
                    best_checkpoint = trainer_state['best_model_checkpoint']
                    # match prefix before /checkpoint of the checkpoint name with the output_dir
                    prefix = best_checkpoint.split("/checkpoint")[0]
                    substitute_prefix = training_args.output_dir.split("/checkpoint")[0]
                    best_checkpoint = best_checkpoint.replace(prefix, substitute_prefix) 
                    logger.info(f"Loading model from {best_checkpoint}")
                    model = RobertaForSequenceClassification.from_pretrained(best_checkpoint, num_labels=num_labels)
        else:
            logger.info("No model found. Using CodeBERT model")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics= lambda x: compute_metrics(x, False, grouped_indices, grouped_labels, pass_idx),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        pred_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = pred_output.predictions
        metrics = pred_output.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics(data_args.predict_suffix, metrics)

        output_predict_file = os.path.join(training_args.output_dir, data_args.predict_suffix)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item_str = "[" + ",".join([str(r) for r in item]) + "]"
                    writer.write(f"{index}\t{item_str}\n")
   
if __name__ == "__main__":
    main()