#!/bin/bash

dataset=../data/kodial_v2_shuffle/custom_toked_lang/kodial-bin
model_dir=../model/kodial_v2_shuffle/custom_token

CUDA_VISIBLE_DEVICES=0 fairseq-train ${dataset} \
--save-dir ${model_dir} \
--tensorboard-logdir $model_dir \
--task translation \
--bpe sentencepiece \
--seed 1 \
--num-workers 3 \
--max-tokens 3584 \
--arch lightconv \
--criterion label_smoothed_cross_entropy \
--update-freq 16 \
--optimizer adam \
--lr '1e-7' \
--clip-norm 0.0 \
--weight-decay 0.001 \
--lr-scheduler cosine \
--lr-shrink 1 \
--min-lr '1e-09' \
--max-lr 0.001 \
--min-loss-scale 0.0001 \
--label-smoothing 0.1 \
--lr-period-updates 20000 \
--warmup-updates 10000 \
--warmup-init-lr '1e-07' \
--adam-betas '(0.9, 0.98)' \
--dropout 0.3 \
--attention-dropout 0.1 \
--relu-dropout 0.0 \
--encoder-embed-dim 512 \
--encoder-ffn-embed-dim 2048 \
--encoder-layers 7 \
--encoder-attention-heads 8 \
--decoder-embed-dim 512 \
--decoder-ffn-embed-dim 2048 \
--decoder-layers 6 \
--decoder-attention-heads 8 \
--share-all-embeddings \
--input-dropout 0.1 \
--encoder-kernel-size-list [3,7,15,31,31,31,31] \
--decoder-kernel-size-list [3,7,15,31,31,31] \
--encoder-glu 1 \
--decoder-glu 1 \
--encoder-conv-type dynamic \
--decoder-conv-type dynamic \
--weight-softmax True \
--weight-dropout 0.1 \
--no-progress-bar \
--no-epoch-checkpoints \
--fp16 \
--log-interval 50;

