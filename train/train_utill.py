from typing import *
from torch.utils.data import DataLoader, RandomSampler

from data.custom_dataset import LineByLineTextDataset, DatasetForTrain

from transformers import DataCollatorForLanguageModeling
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import Seq2SeqTrainingArguments
from train.custom_seq_trainer import Seq2SeqTrainer

def prepare_for_pretraining(
        tokenizer: AutoTokenizer, 
        encoder_input_train: List[str]
        )-> Tuple[LineByLineTextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling]:
    '''
        Arguments:
            tokenizer: AutoTokenizer 
                토크나이저
            encoder_input_train: List[str]
                모델 학습에 사용될 string list

        Return
            Tuple[LineByLineTextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling]

        Summary:
            학습을 위한 dataset에 string을 넣어주고, masking을 위한 collector를 정의해 반환
    '''
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=encoder_input_train,
        block_size=512,
    )
    # set mlm task  DataCollatorForSOP(DataCollatorForLanguageModeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15 # 0.3
    )
    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        data=encoder_input_train[:10],
        block_size=512,
    )

    return dataset, data_collator, eval_dataset

def set_trainer(
        model, 
        data_collator, 
        dataset, 
        eval_dataset,
        epoch: int = 10, 
        batch_size: int = 16, 
        accumalation_step: int = 1,)->Seq2SeqTrainer:
    '''
        Arguments:
            model:  
                학습할 허깅 페이스 모델
            data_collator:
                데이터 수집 방식, making, nsp 등
            dataset, eval_dataset:
                학습 및 검증 데이터
            epoch
                학습할 epoch 수
            batch_size
                학습에 사용할 batch size
            accumalation_step
                학습 도중 적용할 accumalation_step
        Return
            Seq2SeqTrainer

        Summary:
            학습을 위한 Seq2SeqTrainer 설정 후 반환
    '''
     # set training args
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results/mecab', #./
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumalation_step,
        evaluation_strategy = 'steps',
        eval_steps=15000,
        save_steps=15000,
        save_total_limit=2,
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
        eval_dataset=eval_dataset,    
    )

    return trainer

def set_trainer_for_finetuning(
        model, 
        tokenizer,
        dataset, 
        eval_dataset,
        epoch: int = 10, 
        batch_size: int = 16, 
        accumalation_step: int = 1,)->Seq2SeqTrainer:
    '''
        Arguments:
            model, tokenizer:  
                학습할 허깅 페이스 모델 및 토크나이저
            data_collator:
                데이터 수집 방식, making, nsp 등
            dataset, eval_dataset:
                학습 및 검증 데이터
            epoch
                학습할 epoch 수
            batch_size
                학습에 사용할 batch size
            accumalation_step
                학습 도중 적용할 accumalation_step
        Return
            Seq2SeqTrainer

        Summary:
            학습을 위한 Seq2SeqTrainer 설정 후 반환
    '''
     # set training args
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results/fine_tuning/pretrained_only_dialogue30_10epoch', #./
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumalation_step,
        save_strategy = 'epoch',
        evaluation_strategy = 'epoch',
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        load_best_model_at_end=True,
        seed=42,
    )

    # set Trainer class for pre-training
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,    
    )

    return trainer

def prepare_for_finetuning(
        tokenizer: AutoTokenizer, 
        encoder_input_train: List[str],
        decoder_input_train: List[str],
        decoder_output_train: List[str],
        batch_size: int=16,
        )-> Tuple[DataLoader, DataLoader]:
    '''
        Arguments:
            tokenizer: AutoTokenizer 
                토크나이저
            encoder_input_train: List[str]
                모델 학습에 사용될 string list
            decoder_input: List[str]
                디코더에 들어갈 입력
            decoder_output_train: List[str]
                디코더가 출력해야할 라벨(문장)

        Return
            Tuple[DataLoader, DataLoader]

        Summary:
            학습을 위한 dataset을 tokenizing 하고, DataLoader에 담아 반환
    '''
    tokenized_encoder_inputs = tokenizer(list(encoder_input_train), return_tensors="pt", padding=True, 
                            add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
    tokenized_decoder_inputs = tokenizer(list(decoder_input_train), return_tensors="pt", padding=True, 
                        add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)
    tokenized_decoder_ouputs = tokenizer(list(decoder_output_train), return_tensors="pt", padding=True, 
                        add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)
    

    val_tokenized_encoder_inputs = tokenizer(list(encoder_input_train)[:10], return_tensors="pt", padding=True, 
                        add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
    val_tokenized_decoder_inputs = tokenizer(list(decoder_input_train)[:10], return_tensors="pt", padding=True, 
                        add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)
    val_tokenized_decoder_ouputs = tokenizer(list(decoder_output_train)[:10], return_tensors="pt", padding=True, 
                        add_special_tokens=True, truncation=True, max_length=50, return_token_type_ids=False,)

    train_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs, len(encoder_input_train))
    valid_dataset = DatasetForTrain(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs, 10)

    # random_sampler = RandomSampler(train_dataset)

    # train_dataloader = DataLoader(train_dataset, sampler=random_sampler, batch_size=batch_size)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=16)
    
    return train_dataset, valid_dataset

def samples_print(summary, dialogue, ground_truth, amount = 10):
    '''
        Arguments:
            summary: List[str]
                모델이 예측한 요약문
            dialogue: List[str]
                모델이 요약문을 만든 원본 대화문
            ground_truth: List[str]
                모델이 생성해야할 목표
            amount: int
                디코더가 출력해야할 라벨(문장)

        Return
           
        Summary:
            검증 데이터로 모델이 예측한 요약문을 출력
    '''
    print('-'*100)
    print("Dialogue: ", dialogue)
    print("Ground truth: ", ground_truth)
    print("Predicted: ", summary)
    print('-'*100)