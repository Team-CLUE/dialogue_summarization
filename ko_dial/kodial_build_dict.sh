#!/bin/bash

echo "Start build dict for converting fairseq-vocab style.================"
#python kodial_build_dict.py

##TARGET=../data/kodial/custom_toked_lang/dict.txt
#TARGET=../model/sentencepiece/kodial_sp.spieces.vocab
#OUTPUT=../data/kodial/custom_toked_lang/dict.txt
#cut -f1 ${TARGET} | tail -n +4 | sed "s/$/ 100/g" > ${OUTPUT}
## You also need to remove <unk>, <s> and </s> from the sentencepiece dictionary with a dummy count of 100

TARGET=../data/kodial/kogpt2_toked_lang/dict.txt
#TARGET=../model/sentencepiece/kodial_sp.spieces.vocab
OUTPUT=../data/kodial/kogpt2_toked_lang/dict_.txt
cut -f1 ${TARGET} | sed "s/$/ 100/g" > ${OUTPUT}
# You also need to remove <unk>, <s> and </s> from the sentencepiece dictionary with a dummy count of 100

echo "Finished."