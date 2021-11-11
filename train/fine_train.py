from typing import *

import torch
from torch.cuda.amp import autocast, GradScaler

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import nsml
from tqdm import tqdm

from train.train_utill import samples_print

def finetuning(
            model,
            tokenizer, 
            train_loader, 
            valid_loader, 
            epochs: int = 10, 
            accumalation_step: int = 1) -> None:
    '''
        Arguments:
            model, tokenizer: 
                학습할 모델, 주체 및 토크나이저
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
    
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = train_per_epoch(model, train_loader, optimizer, scaler, accumalation_step, device)
        print(f'{e}: train_loss: {train_loss/len(train_loader):.5f}')
        
        scheduler.step()

        model.eval()
        summary, dialogue, ground_truth = valid_per_epoch(model, tokenizer, valid_loader, device)
        # sample print 하기
        samples_print(summary, dialogue, ground_truth, amount = 10)
        
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
        
def valid_per_epoch( 
            model, 
            tokenizer,
            valid_loader, 
            device)->Tuple[List[str], List[str], List[str]]:
    '''
        Arguments:
            model, optimizer, scaler: 
                학습할 모델, 주체, 최적화 방식 및 learning rate 조절 스케쥴러
            valid_loader: torch.utill.data.DataLoader
                검증에 사용할 수 있는 데이터를 loading 해주는 객체
            device: str
                cuda or cpu
                
        Return
            Tuple[List[str], List[str], List[str]]

        Summary:
            검증데이터를 이용해 모델이 생성한 요약문 리스트를 반환
    '''
    dialogue = []
    summary = []
    ground_truth = []
    with torch.no_grad():
        for item in tqdm(valid_loader):
            generated_ids = model.generate(input_ids=item['input_ids'].to(device), 
                            # do_sample=True, 
                            # max_length=50, 
                            # top_p=0.92, #92%로 설정하고 샘플링하기
                            # top_k=0
                            no_repeat_ngram_size=2, 
                            early_stopping=True,
                            max_length=50, 
                            num_beams=5,
                        )  
            for predict_ids, dialogue_ids, label_ids in zip(generated_ids, item['input_ids'], item['labels']):
                result = tokenizer.decode(predict_ids, skip_special_tokens=True)
                index = result.find('.')
                if index != -1:
                    result = result[:index+1]
                dialogue.append(tokenizer.decode(dialogue_ids, skip_special_tokens=True))
                ground_truth.append(tokenizer.decode(label_ids, skip_special_tokens=True))
                summary.append(result)
    
    return summary, dialogue, ground_truth