import pandas as pd
import argparse
import os
import nsml
from nsml import HAS_DATASET, DATASET_PATH
from glob import glob
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from transformers import BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer

from typing import Dict, List, Optional
import os
import pickle
from tqdm import tqdm

class Mydataset(Dataset):
    def __init__(self, encoder_input, len):
        self.encoder_input = encoder_input
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        return item
    
    def __len__(self):
        return self.len

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
    def make_model_input(dataset, is_valid=False, is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = ['sostoken'] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + ' [SEP] ')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : ' [CLS] ' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + ' [SEP] ')

            return encoder_input, decoder_input, decoder_output

def train_data_loader(root_path) :
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

def bind_model(model,types, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model')
        model.save_pretrained(save_dir)
        
        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):      
        if types == 'tokenizer':
            global tokenizer 
            save_dir = os.path.join(dir_name, 'vocab.txt')            
            tokenizer = BertTokenizer(
                vocab_file = save_dir,
                do_basic_tokenize=False,
            )
            print("tokenizer 로딩 완료!")
        else:
            global generate_model
            save_dir = os.path.join(dir_name, 'model')
            generate_model.from_pretrained(save_dir)
            print("model 로딩 완료!")

    def infer(test_path, **kwparser):
        global tokenizer
        global generate_model
        print(tokenizer)
        print(generate_model)

        preprocessor = Preprocess()

        test_json_path = os.path.join(test_path, 'test_data', '*')
        print(f'test_json_path :\n{test_json_path}')
        test_path_list = glob(test_json_path)
        test_path_list.sort()
        print(f'test_path_list :\n{test_path_list}')

        test_json_list = preprocessor.make_dataset_list(test_path_list)
        test_data = preprocessor.make_set_as_df(test_json_list)

        #print(f'test_data:\n{test_data["dialogue"]}')
        encoder_input_test, decoder_input_test = preprocessor.make_model_input(test_data, is_test= True)

        tokenized_encoder_inputs = tokenizer(list(encoder_input_test), return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=256, return_token_type_ids=False,)
        #tokenized_decoder_inputs = tokenizer.tokenize(decoder_input_test, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
        print(tokenized_encoder_inputs['input_ids'][0:10])

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dataset = Mydataset(tokenized_encoder_inputs, len(encoder_input_test))
        dataloader = DataLoader(dataset, batch_size=8)
        summary = []
        with torch.no_grad():
            for item in tqdm(dataloader):
                generated_ids = generate_model.generate(input_ids=item['input_ids'].to(device), max_length=50, num_beams=2)
                for ids in generated_ids:
                    summary.append(tokenizer.decode(ids, skip_special_tokens=True))

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # return list(zip(pred.flatten(), clipped.flatten()))
        prob = [1]*len(encoder_input_test)
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

    #################
    # Load tokenizer & model
    ################# 
    print(torch.__version__)
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    tokenizer = None
    bind_model(model=tokenizer, types='tokenizer', parser=args)
    #tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    #special_tokens_dict = {'additional_special_tokens': ['#@이름#','#@계정#','#@신원#','#@전번#','#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#']}
    #tokenizer.add_special_tokens(special_tokens_dict)
    nsml.load(checkpoint='0', session='nia2012/dialogue/274')

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    config = BartConfig()
    generate_model = BartForConditionalGeneration(config=config)

    bind_model(model=generate_model, types='model', parser=args)
    nsml.load(checkpoint='13', session='nia2012/dialogue/171')
    generate_model.pad_token_id=0
    generate_model.to('cuda:0')

    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        nsml.save(0)
            