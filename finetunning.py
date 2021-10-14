import pandas as pd
import argparse
import os
import nsml
from nsml import HAS_DATASET, DATASET_PATH
from glob import glob
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments

from typing import Dict, List, Optional
import os
import pickle

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
                    train_dialogue.append(''.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
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
                    test_dialogue.append(''.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))

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
    def make_model_input(dataset, is_valid=False, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + 'eostoken')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : 'sostoken' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + 'eostoken')

            return encoder_input, decoder_input, decoder_output

def train_data_loader(root_path) :
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model')
        model.save_pretrained(save_dir)
        
        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):      
        if parser.types == 'tokenizer':
            global tokenizer 
            save_dir = os.path.join(dir_name, 'vocab.txt')            
            tokenizer = BertTokenizer(
                vocab_file = save_dir,
            )
            print("tokenizer 로딩 완료!")
        else:
            global generate_model
            save_dir = os.path.join(dir_name, 'model')
            generate_model.from_pretrained(save_dir)
            print("model 로딩 완료!")

    def infer(test_path, **kwparser):

        # Do not this file
        prob = 1
        summary = 1

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, summary))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    if args.pause :
        nsml.paused(scope=locals())

    train_path_list = train_data_loader(DATASET_PATH)
    train_path_list.sort()

    preprocessor = Preprocess()
    #################
    # Data Load
    #################
    print('-'*10, 'Load data', '-'*10,)
    train_json_list = preprocessor.make_dataset_list(train_path_list)
    train_data= preprocessor.make_set_as_df(train_json_list)
    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_model_input(train_data)
    print('-'*10, 'Load data complete', '-'*10,)

    #################
    # Load tokenizer & model
    ################# 
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    tokenizer = None
    args.types = 'tokenizer'
    bind_model(model=tokenizer, parser=args)

    nsml.load(checkpoint='0', session='nia2012/dialogue/161')
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    args.types = 'model'
    config = BartConfig()
    generate_model = BartForConditionalGeneration(config=config)

    bind_model(model=generate_model, parser=args)
    nsml.load(checkpoint='0', session='nia2012/dialogue/162')

    #################
    # Set dataset and trainer
    #################
    print('-'*10, 'Set dataset and trainer', '-'*10,)
    generate_model.resize_token_embeddings(len(tokenizer))

    # Dataset, Dataloader
    
    
    # set training args
    training_args = TrainingArguments(
        output_dir='./',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=10,
        evaluation_strategy = 'steps',
        save_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        seed=42,
    )
    # set Trainer class for pre-training
    trainer = Trainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=train_dataloader,    
    )
    print('-'*10, 'Set dataset and trainer complete', '-'*10,)
 
    #DONOTCHANGE (You can decide how often you want to save the model)
    #nsml.save(0)