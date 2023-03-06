#!/bin/bash -x
MODEL=microsoft/codebert-base

DATA_DIR=$1
MODEL_DIR=$2
MODEL_CACHE_DIR=$3
TASK=$4
DATA_PATH=$5
TRAIN_FILE_SUFFIX=train.json
VAL_FILE_SUFFIX=val.json
LABELS_SUFFIX=labels_$TASK.txt
WEIGHTS_SUFFIX=weights_$TASK.txt
LABEL_KEY=${TASK}_label

python3 run_seq_classification_partial.py \
    --output_dir $MODEL_DIR \
    --cache_dir $MODEL_CACHE_DIR \
    --model_name_or_path $MODEL \
    --train_file $DATA_DIR/$TRAIN_FILE_SUFFIX \
    --validation_file $DATA_DIR/$VAL_FILE_SUFFIX \
    --sentence1_key prompt \
    --sentence2_key completion \
    --label_key $LABEL_KEY \
    --labels_file $DATA_DIR/$LABELS_SUFFIX \
    --weights_file $DATA_DIR/$WEIGHTS_SUFFIX \
    --data_path $DATA_PATH \
    --grouped_indices_file $DATA_DIR/val_grouped_indices.npy \
    --grouped_labels_file $DATA_DIR/val_grouped_labels.npy \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --load_best_model_at_end \
    --metric_for_best_model top3_3 \
    --logging_first_step \
    --eval_steps 200 \
    --save_steps 200
