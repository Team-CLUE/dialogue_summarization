import os
from glob import glob
from typing import *

from nsml import HAS_DATASET, DATASET_PATH

from data.preprocessing import Preprocess


def train_data_loader(root_path:str)->List[str]:
    '''
        Arguments:
            root_path: str
                JSON 데이터가 들어있는 root 경로
            
        Return
            List[str]

        Summary:
            root 폴더 안의 훈련을 위한 json file들의 경로 리스트를 반환

    '''       
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

def get_data(decoder_start_token: str,
            eos_token: str,
            train_type: str):
    '''
        Arguments:
            decoder_start_token: str 
                디코더 첫 입력으로 들어갈 토큰. 보통 bos token
            eos_token: str
                문장의 끝을 의미하는 end of sentence toeken
            train_type: str
                pretraining or finetuning 학습 타입 선택

        Return
            Union(List[str], List[str]) or
            Union(List[str], List[str], List[str])

        Summary:
            Seq-to-Seq 모델을 학습 시키기 위해 필요한 special token들을 전처리 해주어 반환
    '''
    train_path_list = train_data_loader(DATASET_PATH)
    train_path_list.sort()

    preprocessor = Preprocess(decoder_start_token, eos_token, train_type)

    train_json_list = preprocessor.make_dataset_list(train_path_list)
    train_data= preprocessor.make_set_as_df(train_json_list)
    
    if train_type == 'pretraining':
        encoder_input_train, decoder_output_train = preprocessor.make_input(preprocessor, dataset=train_data)
        return encoder_input_train, decoder_output_train
    elif train_type == 'finetuning':
        encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(preprocessor, dataset=train_data)
        return encoder_input_train, decoder_input_train, decoder_output_train
    
