import os
os.system("export KMER=5")
os.system("export FILE=5_small")
os.system("export TASK=kmer")
os.system("export MODEL_PATH=../5-new-12w-0")
os.system("export DATA_PATH=./DeepSea_data/$FILE")
os.system("export OUTPUT_PATH=./model_checkpoints/$TASK/$FILE")

os.system("python deepssl.py \
    --model_type dnalong \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnadeepsea \
    --do_train \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=10 \
    --per_gpu_train_batch_size=10  \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 1 \
    --deepSeaClassifier $TASK")

os.system("export KMER=5")
os.system("export FILE=5_small")
os.system("export TASK=cls")
os.system("export MODEL_PATH=../5-new-12w-0")
os.system("export DATA_PATH=./DeepSea_data/$FILE")
os.system("export OUTPUT_PATH=./model_checkpoints/$TASK/$FILE")

os.system("python deepssl.py \
    --model_type dnalong \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnadeepsea \
    --do_train \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=10 \
    --per_gpu_train_batch_size=10  \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 1 \
    --deepSeaClassifier $TASK")


os.system("export KMER=6")
os.system("export FILE=6_small")
os.system("export TASK=kmer")
os.system("export MODEL_PATH=../6-new-12w-0")
os.system("export DATA_PATH=./DeepSea_data/$FILE")
os.system("export OUTPUT_PATH=./model_checkpoints/$TASK/$FILE")

os.system("python deepssl.py \
    --model_type dnalong \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnadeepsea \
    --do_train \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=10 \
    --per_gpu_train_batch_size=10  \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 1 \
    --deepSeaClassifier $TASK")

os.system("export KMER=6")
os.system("export FILE=6_small")
os.system("export TASK=cls")
os.system("export MODEL_PATH=../6-new-12w-0")
os.system("export DATA_PATH=./DeepSea_data/$FILE")
os.system("export OUTPUT_PATH=./model_checkpoints/$TASK/$FILE")

os.system("python deepssl.py \
    --model_type dnalong \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnadeepsea \
    --do_train \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=10 \
    --per_gpu_train_batch_size=10  \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 1 \
    --deepSeaClassifier $TASK")