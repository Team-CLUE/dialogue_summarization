import json
from typing import *

import pandas as pd
from pandas.core.frame import DataFrame

import re
from tqdm import tqdm
from soynlp.normalizer import *
from pykospacing import Spacing

class Preprocess:
    def __init__(self, 
            decoder_start_token: str,
            eos_token: str,
            train_type: str
        ) -> None:
        '''
            Arguments:
                decoder_start_token: str 
                    디코더 첫 입력으로 들어갈 토큰. 보통 bos token
                eos_token: str
                    문장의 끝을 의미하는 end of sentence toeken
                train_type: str
                    pretrain or finetuning 학습 타입 선택

            Summary:
                JSON 형태의 데이터를 받아 학습이 가능한 형태로 정제하여 반환하기 위한 class
        '''

        self.decoder_start_token = decoder_start_token
        self.eos_token = eos_token
        self.train_type = train_type

    @staticmethod
    def make_dataset_list(path_list: List[str]) -> List[Dict]:
        '''
            Arguments:
                path_list: List[str]
                    학습을 위한 JSON 데이터가 담겨있는 경로리스트

            Return
                List[Dict]

            Summary:
                학습을 위한 JSON 데이터가 담겨있는 경로리스트 받아, json 데이터를 읽어 리스트에 저장 후 반환
        '''
        json_data_list = []

        for path in path_list:
            with open(path) as f:
                json_data_list.append(json.load(f))

        return json_data_list

    @staticmethod
    def make_set_as_df(train_set, is_train = True) -> DataFrame:
        '''
            Arguments:
                train_set: List[Dict]
                    JSON 데이터를 읽어 만든 Dictionary list
                is_train: Bool
                    훈련을 위한 것이면 summary 까지 추출, 아니면 dialogue만 추출

            Return
                DataFrame

            Summary:
                모델 학습에 필요한 데이터를 Dict에서 추출해 DataFrame 형태로 반환
        '''
        if is_train:
            train_dialogue = []
            train_dialogue_id = []
            train_dialogue_type = []
            train_summary = []
            for topic in train_set:
                for data in topic['data']:
                    train_dialogue_type.append(data['header']['dialogueInfo']['topic'])
                    train_dialogue_id.append(data['header']['dialogueInfo']['dialogueID'])
                    train_summary.append(data['body']['summary'])
                    train_dialogue.append(' '.join([dialogue['utterance'] for dialogue in data['body']['dialogue']]))
                
            train_data = pd.DataFrame(
                {
                    'Category': train_dialogue_type,
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
    def make_input(self, dataset, is_valid=False, is_test = False) -> Tuple[List[str], List[str]] or Tuple[List[str], List[str], List[str]]:
        '''
            Arguments:
                dataset: DataFrame
                    JSON 데이터를 읽어 만든 Dictionary list
                is_valid, is_test: Bool
                    훈련, 검증, 테스트에 따라 문장들 전처리 후 반환

            Return
                Tuple[List[str], List[str]] or
                Tuple[List[str], List[str], List[str]]

            Summary:
                Seq-to-Seq 모델을 학습 시키기 위해 필요한 special token들을 전처리 해주어 반환
        '''
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.decoder_start_token] * len(dataset)
            encoder_input = delete_char(encoder_input)
            encoder_input = delete_others(encoder_input)
            encoder_input = remove_repeat_char(encoder_input)
            encoder_input = spacing_sent(encoder_input)
            
            
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            decoder_input = [self.decoder_start_token] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            encoder_input = delete_char(encoder_input)
            encoder_input = delete_others(encoder_input)
            encoder_input = remove_repeat_char(encoder_input)
            encoder_input = spacing_sent(encoder_input)
            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.decoder_start_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            encoder_input = delete_char(encoder_input)
            encoder_input = delete_others(encoder_input)
            encoder_input = remove_repeat_char(encoder_input)
            encoder_input = spacing_sent(encoder_input)
            
            if self.train_type == 'pretraining':
                return list(encoder_input) + list(decoder_input), decoder_output
            elif self.train_type == 'finetuning':
                return list(encoder_input), list(decoder_input), list(decoder_output)

def delete_char(texts:List[str])->List[str]:
    '''
        Arguments:
            texts: List[str]
                string 리스트

        Return
            List[str]

        Summary:
            학습전 문장에서 불필요한 characters 제거 후 반환
    '''
    preprocessed_text = []
    proc = re.compile(r"[^가-힣a-zA-Z!?@#$%^&*<>()_ +]")
    for text in tqdm(texts):
        text = proc.sub("", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def delete_others(texts:List[str])->List[str]:
    '''
        Arguments:
            texts: List[str]
                string 리스트

        Return
            List[str]

        Summary:
            학습전 문장에서 불필요한 characters 제거 후 반환
    '''
    preprocessed_text = []
    for text in tqdm(texts):
        candidate = ['#@URL#','#@이름#','#@계정#','#@신원#','#@전번#',
                '#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#', '#@시스템#사진#', '#@시스템#검색#',  '#@시스템#지도#', '#@시스템#기타#', '#@시스템#파일#',
                '#@시스템#동영상#', '#@시스템#송금#', '#@시스템#삭제#']
        for cd in candidate:
            text = text.replace(cd, '')
        preprocessed_text.append(text)
    print("deleted unnecessary tokens (E.g. #@URL#)!!")
    return preprocessed_text

def remove_repeat_char(texts):
    preprocessed_text = []
    for text in tqdm(texts):
        text = repeat_normalize(text, num_repeats=2).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def spacing_sent(texts):
    """
    띄어쓰기를 보정합니다.
    """
    spacing = Spacing()
    preprocessed_text = []
    for text in texts:
        text = spacing(text)
        if text:
            preprocessed_text.append(text)
    return preprocessed_text