from typing import *

import torch
from torch.cuda.amp import autocast, GradScaler

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import nsml
from tqdm import tqdm

def finetuning(
            model, 
            train_loader, 
            valid_loader, 
            epochs: int = 10, 
            accumalation_step: int = 1) -> None:
    '''
        Arguments:
            model: 
                학습할 모델, 주체
            train_loader, valid_loader: torch.utill.data.DataLoader
                모델 학습 및 검증에 사용할 수 있는 데이터를 loading 해주는 객체
            epochs: int
                학습할 epoch
            accumalation_step: int
                몇 batch에 한번씩 update 할 것인지

        Return
            
        Summary:
            model을 주어진 데이터로 학습시킨다.
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 200, len(train_loader) * epochs)
    print(train_loader, valid_loader)
    for e in tqdm(range(epochs)):
        train_loss = train_per_epoch(model, train_loader, optimizer, scaler, accumalation_step, device)
        print(f'{e}: train_loss: {train_loss/len(train_loader):.5f}')
        
        scheduler.step()
        #valid_samples = valid_per_epoch()
        # sample print 하기
        
        nsml.save(e)

def train_per_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            accumalation_step, 
            device):
    '''
        Arguments:
            model, optimizer, scaler: 
                학습할 모델, 주체, 최적화 방식 및 learning rate 조절 스케쥴러
            train_loader: torch.utill.data.DataLoader
                모델 학습 및 검증에 사용할 수 있는 데이터를 loading 해주는 객체
            epochs: int
                학습할 epoch
            accumalation_step: int
                몇 batch에 한번씩 update 할 것인지

        Return
            float

        Summary:
            모델을 한 epoch 학습시킨다.
    '''
    batch_loss = 0
    optimizer.zero_grad()
    for step, batch_item in enumerate(tqdm(train_loader)):
        intput = batch_item['input_ids'].to(device)
        intput_atm = batch_item['attention_mask'].to(device)
        decoder_input = batch_item['decoder_input_ids'].to(device)
        decoder_input_atm = batch_item['decoder_attention_mask'].to(device)
        label = batch_item['labels'].to(device)
        
        with autocast():
            loss = model(input_ids=intput, attention_mask=intput_atm, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_input_atm,  labels=label)[0]
            scaler.scale(loss).backward()
            batch_loss += loss.detach().cpu().numpy()
        
        if (step + 1) % accumalation_step == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    return batch_loss
        
def valid_per_epoch():
    print('코딩중')