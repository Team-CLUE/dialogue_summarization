import pandas as pd
import argparse
import os
import nsml
from nsml import HAS_DATASET, DATASET_PATH
from glob import glob
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, BartConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from transformers import LineByLineTextDataset
from tokenizers import BertWordPieceTokenizer 

from typing import Dict, List, Optional
import os
import pickle
import re
from soynlp.normalizer import *
from transformers.utils.dummy_sentencepiece_objects import BarthezTokenizer

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
                    #train_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
                    train_summary.append(data['body']['summary'])

                    prev_pid = data['body']['dialogue'][0]['participantID']
                    utter = f'<{prev_pid}>'
                    
                    for dialogue in data['body']['dialogue']:
                        pid = dialogue['participantID']
                        next_utter = dialogue['utterance']
                        if pid != prev_pid:
                            next_utter = f'</{prev_pid}> <{pid}>' + next_utter 
                        utter += next_utter
                        prev_pid = pid
                    utter += f'</{pid}>'
                    train_dialogue.append(utter)

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
                    #test_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
                    prev_pid = data['body']['dialogue'][0]['participantID']
                    utter = f'<{prev_pid}>'
                    
                    for dialogue in data['body']['dialogue']:
                        pid = dialogue['participantID']
                        next_utter = dialogue['utterance']
                        if pid != prev_pid:
                            next_utter = f'</{prev_pid}> <{pid}>' + next_utter 
                        utter += next_utter
                        prev_pid = pid
                    utter += f'</{pid}>'
                    test_dialogue.append(utter)

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
            decoder_input = ['</s>'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = ['</s>'] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : '<usr>' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + '</s>')

            return list(encoder_input) + list(decoder_input), decoder_output

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
        # tokenizer pretrained 사용
        print("로딩 완료!")

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

def delete_char(texts):
    preprocessed_text = []
    proc = re.compile(r"[^가-힣a-zA-Z/!?@#$%^&*<>()_ +]")
    for text in tqdm(texts):
        text = proc.sub("", text).strip()
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

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data, block_size: int):
        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

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
    encoder_input_train, decoder_output_train = preprocessor.make_model_input(train_data)
    print('-'*10, 'Load data complete', '-'*10,)

    encoder_input_train = delete_char(encoder_input_train)
    encoder_input_train = remove_repeat_char(encoder_input_train)

    #input_train = np.concatenate([list(encoder_input_train), list(decoder_input_train)], axis=0)
    print(len(encoder_input_train))
    print(encoder_input_train[145630:145640])

    #################
    # Load tokenizer
    ################# 
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    special_tokens_dict = {'additional_special_tokens': ['#@URL#','#@이름#','#@계정#','#@신원#','#@전번#','#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#', 
                            '<P01>', '</P01>', '<P02>', '</P02>', '<P03>', '</P03>', '<P04>', '</P04>', '<P05>', '</P05>', '<P06>', '</P06>',
                            '<P07>', '</P07>', '<P08>', '</P08>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print('-'*10, 'Load tokenizer complete', '-'*10,)
    
    config = BartConfig().from_pretrained('gogamza/kobart-summarization')
    model = BartForConditionalGeneration(config=config)
    
    if args.pause :
        nsml.paused(scope=locals())

    #################
    # Set dataset and trainer
    #################
    print('-'*10, 'Set dataset and trainer', '-'*10,)
    model.resize_token_embeddings(len(tokenizer))
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=encoder_input_train,
        block_size=512,
    )
    v_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=encoder_input_train[:50],
        block_size=512,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    # set training args
    training_args = Seq2SeqTrainingArguments(
        output_dir='./',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=10,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit=1,
        fp16=True,
        load_best_model_at_end=True,
        seed=42,
    )
    # set Trainer class for pre-training
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=v_dataset,    
    )
    print('-'*10, 'Set dataset and trainer complete', '-'*10,)

    #################
    # Start pretraining
    #################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('-'*10, 'Start pretraining:\t', device, '-'*10,)
    bind_model(model=model, parser=args)
    
    for epoch in range(15):
        trainer.train()
        nsml.save(epoch)
        print('-'*10, '저장완료!', '-'*10,)

    print('-'*10, 'Pretraing complete', '-'*10,)
    