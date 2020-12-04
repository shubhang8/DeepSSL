# DeepSSL
* Enviroment requirements:
python=3.6

# data

http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz

# Step to setup DNABert

https://github.com/jerryji1993/DNABERT

## Simple version:

1. create env
conda create -n dnabert python=3.6
conda activate dnabert

2. install package

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt

3. download pretrained models to DeepSSL/DNABert/.

DNABert3:https://northwestern.box.com/s/s492dj5g2wwotdh40v9uv5gwikqi246q
DNABert4:https://northwestern.box.com/s/rmmepi2upskgob4fgeohdwh1r5w37oqo
DNABert5:https://northwestern.box.com/s/6wjib1tnnt7efj5yzmxc3da0so800c6c
DNABert6:https://northwestern.box.com/s/g8m974tr86h0pvnpymxq84f1yxlhnvbi



4. fine-tunning (use DNABert-6 as an example)

unzip 6-new-12w-0.zip

cd examples

export KMER=6
export MODEL_PATH=../6-new-12w-0
export DATA_PATH=sample_data/ft/prom-core/$KMER
export OUTPUT_PATH=./ft/prom-core/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 75 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
