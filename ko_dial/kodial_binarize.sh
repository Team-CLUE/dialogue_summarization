#!/bin/bash

root=kodial

src=dialog
tgt=summary

#TARGET=../data/kodial/kogpt2_toked
#TARGET=../data/kodial/kogpt2_toked_lang
#TARGET=../data/kodial/custom_toked_lang
#TARGET=../data/kodial/kogpt2_toked_lang
TARGET=../data/kodial_v2_shuffle/custom_toked_lang
# Please extract the dict.txt from pretrained sentencepiece tokenizer / run python kodial_build_dict.py

fairseq-preprocess \
  --source-lang "dialog" \
  --target-lang "summary" \
  --trainpref "${TARGET}/${root}.train" \
  --validpref "${TARGET}/${root}.validation" \
  --testpref "${TARGET}/${root}.test" \
  --destdir "${TARGET}/kodial-bin/" \
  --srcdict "${TARGET}/dict.txt" \
  --tgtdict "${TARGET}/dict.txt" \
  --workers 60 ;
#====================================================
#fairseq-preprocess \
#  --source-lang "dialog" \
#  --target-lang "summary" \
#  --trainpref "${TARGET}/${root}.train" \
#  --validpref "${TARGET}/${root}.validation" \
#  --testpref "${TARGET}/${root}.test" \
#  --destdir "${TARGET}/kodial-bin/" \
#  --joined-dictionary \
#  --workers 60 ;
#====================================================
#  --srcdict "${TARGET}/dict.txt" \
#  --tgtdict "${TARGET}/dict.txt" \



#root=samsum
##bin=$root/bin
#src=dialog
#tgt=summary
#gpt2=../gpt2_bpe
#
##mkdir $bin
#TASK=samsum
#fairseq-preprocess \
#  --source-lang "dialog" \
#  --target-lang "summary" \
#  --trainpref "${TASK}/${root}.train.bpe" \
#  --validpref "${TASK}/${root}.val.bpe" \
#  --testpref "${TASK}/${root}.test.bpe" \
#  --destdir "${TASK}-bin/" \
#  --workers 60 \
#  --srcdict "${gpt2}/dict.txt" \
#  --tgtdict "${gpt2}/dict.txt";