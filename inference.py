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
import re
from tqdm import tqdm

class Mydataset(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
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
            #decoder_input = ['</s>'] * len(dataset)
            decoder_input = ['<usr>'] * len(dataset)
            return encoder_input, decoder_input

        elif is_valid:
            encoder_input = dataset['dialogue']
            #decoder_input = ['</s>'] * len(dataset)
            decoder_input = ['<usr>'] * len(dataset)
            decoder_output = dataset['summary'].apply(lambda x: str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output

        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : '<usr>' + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + '</s>')

            return encoder_input, decoder_input, decoder_output

def train_data_loader(root_path) :
    train_path = os.path.join(root_path, 'train', 'train_data', '*')
    pathes = glob(train_path)
    return pathes

def bind_model(model, tokenizer, types, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model')
        model.save_pretrained(save_dir)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):      
        #global generate_model
        print(model)
        save_dir = os.path.join(dir_name, 'model/pytorch_model.bin')
        state_dict = torch.load(save_dir) 
        model.load_state_dict(state_dict)
        #model.from_pretrained(save_dir)
        print("model 로딩 완료!")

    def infer(test_path, **kwparser):
        #global tokenizer
        #global generate_model
        #print(tokenizer)
        #print(generate_model)

        preprocessor = Preprocess()

        test_json_path = os.path.join(test_path, 'test_data', '*')
        #print(f'test_json_path :\n{test_json_path}')
        test_path_list = glob(test_json_path)
        test_path_list.sort()
        #print(f'test_path_list :\n{test_path_list}')

        test_json_list = preprocessor.make_dataset_list(test_path_list)
        test_data = preprocessor.make_set_as_df(test_json_list)
        test_id = test_data['dialogueID']

        #print(f'test_data:\n{test_data["dialogue"]}')
        encoder_input_test, decoder_input_test = preprocessor.make_model_input(test_data, is_test= True)
        
        ######################
        encoder_input_test = delete_char(encoder_input_test)

        tokenized_encoder_inputs = tokenizer(list(encoder_input_test), 
                return_tensors="pt", 
                add_special_tokens=True, 
                padding=True, truncation=True, 
                max_length=256, 
                return_token_type_ids=False,)
        #tokenized_decoder_inputs = tokenizer.tokenize(decoder_input_test, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
        #print(tokenized_encoder_inputs['input_ids'][0:10])

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        dataset = Mydataset(tokenized_encoder_inputs, test_id, len(encoder_input_test))
        
        dataloader = DataLoader(dataset, batch_size=128)
        
        summary = []
        text_ids = []
        with torch.no_grad():
            for item in tqdm(dataloader):
                text_ids.extend(item['ID'])
                generated_ids = generate_model.generate(input_ids=item['input_ids'].to(device), 
                                # do_sample=True, 
                                # max_length=50, 
                                # top_p=0.92, #92%로 설정하고 샘플링하기
                                # top_k=0
                                no_repeat_ngram_size=2, 
                                early_stopping=True,
                                max_length=50, 
                                num_beams=5,
                            )  
                for ids in generated_ids:
                    result = tokenizer.decode(ids, skip_special_tokens=True)
                    index = result.find('.')
                    if index != -1:
                        result = result[:index+1]
                    summary.append(result)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # return list(zip(pred.flatten(), clipped.flatten()))
        
        return list(zip(text_ids, summary))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def delete_char(texts):
    preprocessed_text = []
    proc = re.compile(r"[^가-힣a-zA-Z!?@#$%^&*<>()_ +]")
    for text in tqdm(texts):
        text = proc.sub("", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

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
    #tokenizer = None
    #bind_model(model=tokenizer, types='tokenizer', parser=args)
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    special_tokens_dict = {'additional_special_tokens': ['#@URL#','#@이름#','#@계정#','#@신원#','#@전번#',
                '#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#', '#@시스템#사진', '#@시스템#검색',  '#@시스템#지도#', '#@시스템#기타#', '#@시스템#파일#',
                '#@시스템#동영상#', '#@시스템#송금#', '#@시스템#삭제#']}
    tokenizer.add_special_tokens(special_tokens_dict)
    #nsml.load(checkpoint='0', session='nia2012/dialogue/274')

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    config = BartConfig().from_pretrained('gogamza/kobart-summarization')
    # config.d_model = 1024
    # config.decoder_attention_heads = 16
    # config.decoder_ffn_dim = 4096
    # config.decoder_layers = 10

    # config.encoder_attention_heads = 16
    # config.encoder_ffn_dim = 4096
    # config.encoder_layers = 10

    generate_model = BartForConditionalGeneration(config=config)
    generate_model.resize_token_embeddings(len(tokenizer))  

    #%%
    bind_model(model=generate_model, tokenizer=tokenizer, types='model', parser=args)
    nsml.load(checkpoint=0, session='nia2012/final_dialogue/84')
    generate_model.to('cuda:0')
    
    # score = Rouge()
    # test = ['#@시스템#사진# #@이름#이는 매일매일이 지옥인가부다.. 번호 저장을안해서 #@이름#로 뜨네 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 마자 지난주에 저러더라 못살아 증말 ㅋㅋㅋㅋㅋㅋㅋ어쩌냐 진짜 결혼 잘못한 케이스같은데... 대충만 들었어도 엉망이었잖아 응... 어린나이에 너무속상하다 그니까ㅠㅠ<usr>',
    #    '난 악몽꿨던거 정확히 기억은 안나는데 응응 누가 날 죽이려고 그랬어 나는 계속 도망다니고 헐... 너도 스트레스 많이 받아서 그런가봐 응응 그래서 내가 살려달라고 그랬더니 그 사람이 이거 꿈이야 헐 그게 더 무서워<usr>',
    #    '이겜 강아지 고기랑 사료만 먹는듯? 응 글타데 여러마리일수록 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 많이먹는데 강아지키우면서 책임감이 생겻구만 그런 강아지를 너는 뺏어왓지 ㅋㅋㅋㅋㅋ 내가 새로운주인이다 그럼 안돼<usr>',
    #    '토익점수 왜케안나왔지 듣기를 올려봐 그니까 듣기를 더해야겟어 웅 열심히해야하지 독해는 올랐는데 듣기가 떨어지네 요번 파트가 점수를 엄청깎았다는데 그래서그런가봐 점만 올리자!!!!!!!!!!!!!!!!!!!! 듣기 점 올려야되서 열심히해야하지!!!!!!!!!!!!']

    # labels = ['대충만 들었는데도 엉망이었다고 어린 나이에 결혼을 잘 못한 케이스 같아서 너무 속상하다고 한다.',
    #    '악몽을 꾼 것이 정확히 기억이 나진 않지만 계속 도망다녔다고 하며 스트레스를 많이 받아서라고 한다.',
    #    '자신들이 하고 있는 강아지를 키우며 책임감을 키우는 게임에 대해 대화한다.']


    # tokenized = tokenizer(test, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=256, return_token_type_ids=False,)
    
    # dataset = Mydataset(tokenized, [1,2,3,4] ,len(test))
    # dataloader = DataLoader(dataset, batch_size=128)
    
    # summary = []
    # text_ids = []
    # with torch.no_grad():
    #     for item in tqdm(dataloader):
    #         text_ids.extend(item['ID'])
    #         generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'), 
    #                         no_repeat_ngram_size=2, 
    #                         early_stopping=True, 
    #                         max_length=50, 
    #                         num_beams=5,
    #                     )
    #         for ids in generated_ids:
    #             result = tokenizer.decode(ids, skip_special_tokens=True)
    #             index = result.find('.')
    #             if index != -1:
    #                 result = result[:index+1]
    #             summary.append(result)
    
    # for idx, te in enumerate(summary):
    #     print(idx, te)

    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        nsml.save(0)
            



            