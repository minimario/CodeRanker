# Import argument parsers
from argparse import ArgumentParser
from run_seq_classification_partial import (
    ModelArguments,
    DataTrainingArguments,
    CustomTrainer,
)
import wandb
import numpy as np
import os

from transformers import (
    AutoConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    EvalPrediction,
    T5ForConditionalGeneration,
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from t5_encoder_classifier import T5EncoderForSequenceClassification


def setup_wandb(model_args, data_args, training_args):
    all_args = {**vars(model_args), **vars(data_args), **vars(training_args)}
    wandb.init(project="huggingface", entity="codegen", config=all_args)
    return


from run_codebert_data import preprocess_function

# def preprocess_function(tokenizer, example: dict):
#     print(type(example))
#     input_id = tokenizer(
#         example["code"], truncation=True, padding="max_length", max_length=512
#     )
#     attention_mask = [1] * len(input_id["input_ids"])
#     return {
#         "input_ids": input_id["input_ids"],
#         "attention_mask": attention_mask,
#         "labels": 1,
#     }


def compute_metrics(eval_pred: EvalPrediction):
    # accuracy, precision, recall, f1
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": (predictions == labels).mean(),
        "accuracy_sklearn": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def process_last_checkpoint(training_args):
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            training_args.do_train
            and last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def main():
    # Initialize parser
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)
    print(training_args)

    # Setup wandb
    training_args.report_to = ["wandb"]
    setup_wandb(model_args, data_args, training_args)

    # Make sure things are deterministic
    set_seed(training_args.seed)

    # Load the model and tokenizer
    # model_args.model_name_or_path = "microsoft/codebert-base"
    model_args.model_name_or_path = "Salesforce/codet5-large"
    t5_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    t5_config.num_labels = 2
    t5_model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    model = T5EncoderForSequenceClassification(t5_model.encoder, t5_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer
    )
    print("Model loaded!")

    # Load the datasets
    if training_args.do_eval or training_args.do_predict:
        eval_dataset = Dataset.load_from_disk(data_args.validation_file)
        print("Eval dataset loaded!")
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(tokenizer, max_seq_length, x),
            batched=True,
            desc="Running tokenizer on eval dataset",
            load_from_cache_file=False,
            # num_proc=os.cpu_count(),
            num_proc=5,
        )
    # dataset: [input_ids, attention_mask, label]
    if training_args.do_eval or training_args.do_predict:
        print("Eval dataset stats: ")
        print("Total number of samples: ", len(eval_dataset))
        print("Percentage of 0's: ", eval_dataset["label"].count(0) / len(eval_dataset))
        print("Percentage of 1's: ", eval_dataset["label"].count(1) / len(eval_dataset))

    if training_args.do_train:
        train_dataset = Dataset.load_from_disk(data_args.train_file)
        print("Train dataset loaded!")
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(tokenizer, max_seq_length, x),
            batched=True,
            desc="Running tokenizer on train dataset",
            load_from_cache_file=False,
            # num_proc=os.cpu_count(),
            num_proc=5,
        )

    if training_args.do_train:
        print("Train dataset stats: ")
        print("Total number of samples: ", len(train_dataset))
        print(
            "Percentage of 0's: ", train_dataset["label"].count(0) / len(train_dataset)
        )
        print(
            "Percentage of 1's: ", train_dataset["label"].count(1) / len(train_dataset)
        )

    # Now, train
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset if training_args.do_train else None,
        # train_dataset=eval_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )
    trainer.set_weights([0.5, 0.5])

    if training_args.do_train:
        last_checkpoint = process_last_checkpoint(training_args)
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        for i in range(1):
            trainer._load_from_checkpoint(
                # os.path.join(training_args.output_dir, f"checkpoint-{i*1000}")
                os.path.join(training_args.output_dir, f"checkpoint-200")
            )

            # make output directory
            if not os.path.exists(os.path.join(training_args.output_dir, "results")):
                os.makedirs(os.path.join(training_args.output_dir, "results"))

            # predict on eval_dataset
            predictions, labels, metrics = trainer.predict(eval_dataset)
            #     eval_dataset.select(list(range(1000)))
            # )
            trainer.log_metrics(f"eval-{i*1000}", metrics)
            trainer.save_metrics(f"results/eval-{i*1000}", metrics)
            # save predictions
            np.save(
                os.path.join(
                    os.path.join(training_args.output_dir, "results"),
                    f"predictions_{i*1000}.npy",
                ),
                predictions,
            )
            np.save(
                os.path.join(
                    os.path.join(training_args.output_dir, "results"),
                    f"labels_{i*1000}.npy",
                ),
                labels,
            )


if __name__ == "__main__":
    main()
