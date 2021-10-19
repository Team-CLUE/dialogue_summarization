import pandas as pd
import numpy as np
import os
from glob import glob
import json
import re
from tqdm.auto import tqdm
from soynlp.normalizer import *

class Preprocess:

    @staticmethod
    def make_dataset_list(path_list):
        json_data_list = []

        for path in path_list:
            with open(path) as f:
                json_data_list.append(json.load(f))

        return json_data_list

    @staticmethod
    def make_set_as_df(train_set, is_train = True):

        if is_train:
            train_dialogue = []
            train_dialogue_id = []
            train_summary = []
            for topic in train_set:
                for data in topic['data']:
                    train_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    train_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
                    train_summary.append(data['body']['summary'])

            train_data = pd.DataFrame(
                {
                    'dialogueID': train_dialogue_id,
                    'dialogue': train_dialogue,
                    'summary': train_summary
                }
            )
            return train_data

        else:
            test_dialogue = []
            test_dialogue_id = []
            for topic in train_set:
                for data in topic['data']:
                    test_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    test_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))

            test_data = pd.DataFrame(
                {
                    'dialogueID': test_dialogue_id,
                    'dialogue': test_dialogue
                }
            )
            return test_data

    @staticmethod
    def train_valid_split(train_set, split_point):
        train_data = train_set.iloc[:split_point, :]
        val_data = train_set.iloc[split_point:, :]

        return train_data, val_data

    @staticmethod
    def make_tokenizer_input(dataset, is_valid=False, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = ['<s>'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = ['<s>'] * len(dataset)
            #decoder_output = dataset['summary'].apply(lambda x: str(x) + ' [SEP] ')
            decoder_output = dataset['summary'].apply(lambda x: str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            #decoder_input = dataset['summary'].apply(lambda x : ' [CLS] ' + str(x) + ' [SEP] ')
            decoder_input = dataset['summary'].apply(lambda x : '<s>' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + '</s>')

            return list(encoder_input) + list(decoder_input), decoder_output
        
    @staticmethod
    def make_model_input(dataset, is_valid=False, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = ['<s>'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = ['<s>'] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : '<s>' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output
        
def delete_char(texts):
    preprocessed_text = []
    proc = re.compile(r"[^가-힣a-zA-Z!?@#$%^&*<>()_ +]")
    for text in tqdm(texts):
        text = proc.sub("", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def spacing_sent(texts):
    """
    띄어쓰기를 보정합니다.
    """
    spacing = Spacing()
    preprocessed_text = []
    for text in tqdm(texts):
        text = spacing(text)
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def remove_repeat_char(texts):
    preprocessed_text = []
    for text in tqdm(texts):
        text = repeat_normalize(text, num_repeats=2).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text