#!/bin/bash

usage() { echo "Usage: $0 [-d <true|false>] [-t <kogpt2|kobart|custom>] [-d whether do data preprocess] [-t tokenizer selection]" 1>&2; exit 1; }


while getopts ":hd:t:" option;
do
  case "${option}" in
    t)
      TOK=${OPTARG}
      if [[ "${TOK}" != kobart && "${TOK}" != kogpt2 && "${TOK}" != custom ]]; then
        echo "tokenizer needs to be either kogpt2 or kobart or custom, $TOK found instead."
        usage
      fi
      ;;
    d)
      PRE=${OPTARG}
      if [[ "${PRE}" != false && "${PRE}" != true ]]; then
        echo "data preprocess needs to be either false or true, $PRE found instead."
        usage
      fi
      ;;
    h | *)
      echo "Invalid parameter"
      usage
      exit 0 ;;
  esac
done
shift $((OPTIND-1))

if [ -z "${PRE}" ] || [ -z "${TOK}" ]; then
    usage
fi

if [ $PRE == true ]; then
  URL="https://drive.google.com/uc?id=1xUQdXVGtmk7iD4oftD6LJKFcj56u7P3z"

  GZ=data.jsonl

  root=../data
  src=dialog
  tgt=summary

  corpusname=kodial
  orig=$root/$corpusname/orig

  mkdir $orig

  echo "Downloading data from ${URL}..."
  cd $orig
  gdown $URL -O $GZ

  if [ -f $GZ ]; then
      echo "Data successfully downloaded."
  else
      echo "Data not successfully downloaded."
      exit
  fi

  cd ../../../ko_dial
  echo "Data align pre-processing ..."
  python kodial_data_align.py
fi
#########################################################

if [ $TOK == kogpt2 ]; then
#  TOK_PATH=../data/kodial/kogpt2_toked
  TOK_PATH=../data/kodial/kogpt2_toked_lang
  mkdir $TOK_PATH
  echo "KoGPT2 tokenize processing ..."
  python kodial_tokenizer.py --tokenizer $TOK
fi

if [ $TOK == custom ]; then
#  TOK_PATH=../data/kodial/kogpt2_toked
#  TOK_PATH=../data/kodial/custom_toked_lang
  TOK_PATH=../data/kodial_v2_shuffle/custom_toked_lang
  mkdir $TOK_PATH
  echo "Customed tokenize processing ..."
  python kodial_tokenizer.py --tokenizer $TOK
fi


if [ $TOK == kobart ]; then
  TOK_PATH=../data/kodial/kobart_toked

  mkdir $TOK_PATH
  echo "KoBART tokenize processing ..."
#  python kodial_tokenizer.py --tokenizer $TOK
fi


