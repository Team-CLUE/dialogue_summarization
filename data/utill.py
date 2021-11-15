import logging
import os
from glob import glob
from typing import Tuple, List
from data.preprocessing import Preprocess

DATASET_PATH = "./dataset/kor-dialouge-data"
LOG_PATH = "log/"

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
    train_path = os.path.join(root_path, 'Training', '*', '*')
    pathes = glob(train_path)
    return pathes


def get_data(
    decoder_start_token: str,
    eos_token: str,
    train_type: str
)->Tuple[List[str], List[str], List[str]]:
    '''
        Arguments:
            decoder_start_token: str 
                디코더 첫 입력으로 들어갈 토큰. 보통 bos token
            eos_token: str
                문장의 끝을 의미하는 end of sentence toeken
            train_type: str
                pretraining or finetuning 학습 타입 선택

        Return
            Tuple[List[str], List[str], List[str]]

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


def get_logger(logging_path : str = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Set default logging file
    if logging_path is None:
        logging_path = "output.log"

    # Set logging path
    logging_path = os.path.join(LOG_PATH, logging_path)
    
    # Check file exist
    if os.path.exists(logging_path):
        pass

    # Set formatter
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    # print log to sys.out
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # print log to logging file
    file_handler = logging.FileHandler(logging_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger