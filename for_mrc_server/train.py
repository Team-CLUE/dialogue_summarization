from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BartTokenizer, AutoTokenizer
from transformers import BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
from preprocessing import *
import os
from glob import glob
import re
from tqdm.auto import tqdm
from soynlp.normalizer import *
import torch
from tokenizers import BertWordPieceTokenizer 
import pandas as pd
from datasets import load_dataset, load_metric

class Mydataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item
    
    def __len__(self):
        return self.len

if __name__=='__main__':
    learning_tokenizer = False
    vocab_dir = './vocab/vocab.txt'
    
    preprocess = Preprocess()
    
    # 데이터 불러오기
    train_path_list = './new_data/train_data.csv'#glob('./Train/*')
    train_data = pd.read_csv(train_path_list)
    # train_json_list = preprocess.make_dataset_list(train_path_list)
    # train_data= preprocess.make_set_as_df(train_json_list)
    train_data.rename(columns={'카테고리': 'category', '대화 내용': 'dialogue', '대화 내용 + 화자 타입': 'dialogue_type', '대화 요약문':'summary'}, inplace =True)
    encoder_input_train, decoder_input_train, decoder_output_train = preprocess.make_model_input(train_data)
    
    test_json_path = './Valid/식음료.json'
    test_path_list = glob(test_json_path)
    test_path_list.sort()

    test_json_list = preprocess.make_dataset_list(test_path_list)
    test_data = preprocess.make_set_as_df(test_json_list)

    encoder_input_test, decoder_input_test, decoder_output_test = preprocess.make_model_input(test_data)

    # 전처리
    encoder_input_train = delete_char(encoder_input_train)
    encoder_input_train = remove_repeat_char(encoder_input_train)

    encoder_input_test = delete_char(encoder_input_test)
    encoder_input_test = remove_repeat_char(encoder_input_test)
    print(encoder_input_train[:2])
    print(encoder_input_train[-2:])
    
    # 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    special_tokens_dict = {'additional_special_tokens': ['#@이름#','#@계정#','#@신원#','#@전번#','#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    ##print(tokenizer.convert_tokens_to_ids('<s>'), tokenizer.convert_tokens_to_ids('</s>'), tokenizer.convert_tokens_to_ids('<pad>'), tokenizer.convert_ids_to_tokens(2))
    

    # 데이터 만들기
    tokenized_encoder_inputs = tokenizer(list(encoder_input_train), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
    tokenized_decoder_inputs = tokenizer(list(decoder_input_train), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)
    tokenized_decoder_ouputs = tokenizer(list(decoder_output_train), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)

    valid_tokenized_encoder_inputs = tokenizer(list(encoder_input_test), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
    valid_tokenized_decoder_inputs = tokenizer(list(encoder_input_test), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)
    valid_tokenized_decoder_ouputs = tokenizer(list(encoder_input_test), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)

    encoder_inputs_dataset = Mydataset(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs, len(encoder_input_train))
    valid_encoder_inputs_dataset = Mydataset(valid_tokenized_encoder_inputs, valid_tokenized_decoder_inputs, valid_tokenized_decoder_ouputs, len(encoder_input_test))
    #print(tokenized_encoder_inputs['input_ids'][0])
    
    # 모델
    '''
    config.d_model = 1024
    config.encoder_ffn_dim = 512
    config.decoder_ffn_dim = 512
    config.encoder_attention_heads = 8
    config.decoder_attention_heads = 8
    config.max_position_embeddings = 512
    config.encoder_layers = 6
    config.decoder_layers = 6
    '''
    config = BartConfig()
    config.d_model = 1024
    config.encoder_ffn_dim = 1024
    config.decoder_ffn_dim = 1024
    config.encoder_attention_heads = 8
    config.decoder_attention_heads = 8
    config.max_position_embeddings = 1024
    config.encoder_layers = 8
    config.decoder_layers = 8
    config.pad_token_id=3
    config.decoder_start_token_id=1
    config.eos_token_id=1
    config.bos_token_id=0
    model = BartForConditionalGeneration(config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(config)
    
    # 훈련
    training_args = Seq2SeqTrainingArguments(
        output_dir='./bart_prerprocess_without_pretraining_v6/',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=10,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit=1,
        fp16 =True,
        load_best_model_at_end=True,
        seed=42,
    )
    # set Trainer class for pre-training
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoder_inputs_dataset,
        eval_dataset=valid_encoder_inputs_dataset,
    )
    
    trainer.train()