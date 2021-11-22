from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def finetuning(
            model, 
            train_set, 
            args : Seq2SeqTrainingArguments = None,
    ) -> Seq2SeqTrainer:
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
            args : Seq2SeqTrainingArguments
                트레이너 class arguments

        Return
            없음
        Summary:
            model을 주어진 데이터로 학습시킨다.
    '''
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = args,
        train_dataset=train_set,
        eval_dataset=None,
    )

    return 

    
