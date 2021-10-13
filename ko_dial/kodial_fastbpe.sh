#!/bin/bash

#URL="https://drive.google.com/uc?id=1xUQdXVGtmk7iD4oftD6LJKFcj56u7P3z"
#
#GZ=data.jsonl
#
#root=../data
#src=dialog
#tgt=summary
#
#corpusname=kodial
#orig=$root/$corpusname/orig
#output=$root/$corpusname/fastBPE
#
#mkdir -p $orig $corpusname $output
#
#echo "Downloading data from ${URL}..."
#cd $orig
#gdown $URL -O $GZ
##wget --user-agent="Mozilla" "$URL"
#
#if [ -f $GZ ]; then
#    echo "Data successfully downloaded."
#else
#    echo "Data not successfully downloaded."
#    exit
#fi
#
#cd ../../../ko_dial
#echo "Data align pre-processing ..."
#python kodial_data_align.py
#########################################################

TOK_PATH=../data/kodial/kogpt2_toked

mkdir $TOK_PATH
echo "KoGPT2 tokenize processing ..."
python kodial_tokenizer.py

###########################################################
## pip install fastBPE subword_nmt
## subword-nmt bpe
#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git
#
#BPEROOT=subword-nmt/subword_nmt
#TRAIN=$root/$corpusname/samsum.train.$src
#BPE_CODE=$output/code
#BPE_TOKENS=30000
#
#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
#
#for L in $src $tgt; do
#    for f in train.$L valid.$L test.$L; do
#        echo "apply_bpe.py to ${f}..."
#        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $root/$corpusname/samsum.$f > $output/samsum.$f.bpe
#    done
#done

##############################################################
#BPE
#codes=30000
#target=../data/$corpusname
#output_bpe=$output/bpe.30k
#mkdir $output_bpe
#
#echo 'Cloning fastBPE repository (for BPE pre-processing)...'
#git clone https://github.com/glample/fastBPE.git
#pushd fastBPE
#g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
#popd
#fastBPE/fast learnbpe $codes $target/$corpusname.train.$src $target/$corpusname.train.$tgt > $output_bpe/codes
#for split in {train,valid,test}; do for lang in {dialog,summary}; do fastBPE/fast applybpe $output_bpe/samsum.$split.bpe.$lang $target/samsum.$split.$lang $output_bpe/codes; done; done

#echo 'Get train vocabulary.....'


