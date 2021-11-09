from typing import *

from data.custom_dataset import LineByLineTextDataset

from transformers import DataCollatorForLanguageModeling
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

def prepare_for_pretraining(
        tokenizer: AutoTokenizer, 
        encoder_input_train: List[str]
        ):#-> Union(LineByLineTextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling):
    '''
        Arguments:
            tokenizer: AutoTokenizer 
                토크나이저
            encoder_input_train: List[str]
                모델 학습에 사용될 string list

        Return
            Union(LineByLineTextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling)

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
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
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
        output_dir='./',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumalation_step,
        evaluation_strategy = 'steps',
        eval_steps=15000,
        save_steps=15000,
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
        eval_dataset=eval_dataset,    
    )

    return trainer