#!/bin/bash
#SBATCH -p gpu-mono
#SBATCH -n 1 
#SBATCH --gres=gpu:1
#SBATCH --mem=0 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com
#SBATCH --output=/tmp-network/user/%u/tmp/%j


RAW_DIR=/tmp-network/user/mgalle/tagged/
PREFIX=wiki.train-400000.raw
name=roberta-base-tok-fixed
pruning=$1
model=roberta-small-baseline-lth-local-$pruning
DATA_DIR=${RAW_DIR}/$name/

mname=${name}/${model}
MODEL_DIR=/tmp-network/user/vnikouli/model/${mname}
init_checkpoint=$MODEL_DIR/../roberta-small-baseline/checkpoint10.pt
final_checkpoint=$MODEL_DIR/../roberta-small-baseline/checkpoint100.pt
mkdir -p ${MODEL_DIR}
TB_DIR=/tmp-network/user/vnikouli/tensorboard_logs/${mname}
echo $MODEL_DIR
mkdir -p ${TB_DIR}
source ~/miniconda3/etc/profile.d/conda.sh
FAIRSEQ=/tmp-network/user/vnikouli/Projects/LTH/fairseq-LTH/fairseq
conda activate $FAIRSEQ/../../cenv 


cp $0 $MODEL_DIR/

which python
echo $SLURM_JOBID
echo "LOG AT ${MODEL_DIR}/train.${SLURM_JOBID}.log"


fairseq-train ${DATA_DIR}/bin-tokens --task masked_lm \
    --save-dir ${MODEL_DIR} \
    --init-checkpoint $init_checkpoint \
    --final-checkpoint $final_checkpoint \
    --arch roberta_base_lth \
    --pruning $pruning --fp16 \
    --encoder-embed-dim 512 --encoder-attention-heads 4 --encoder-ffn-embed-dim 1024 --encoder-layers 6 \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 --warmup-updates 10000 --total-num-update 125000 --max-epoch 50 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-tokens 4096 --update-freq 16 --ddp-backend=no_c10d \
    --max-update 125000 --log-interval 100 --log-format simple  \
    --criterion masked_lm --tokens-per-sample 512 --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir ${TB_DIR} &> ${MODEL_DIR}/train.${SLURM_JOBID}.log



