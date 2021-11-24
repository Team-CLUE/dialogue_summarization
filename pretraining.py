import argparse
import torch

from data.utill import get_data

from models.toeknizers import get_tokenizer
from models.bart import get_bart_model
from train.train_utill import prepare_for_pretraining, set_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--dtype', type=str, default='pretraining')
    args = parser.parse_args()
    # os.system("wandb login auth코드")

    if args.tokenizer == 'mecab':
        # [CLS]: 2, [SEP]: 3
        decoder_strat_token = ''
        end_of_sentence = ''
    else:
        args.tokenizer = 'gogamza/kobart-summarization'
        decoder_strat_token = '<usr>'
        end_of_sentence = '</s>'

    #################
    # Load data
    ################# 
    print('-'*10, 'Load data', '-'*10,)
    datas = get_data(decoder_strat_token, end_of_sentence, args.dtype) # -> finetuning
    if len(datas) == 2:
        encoder_input_train, decoder_output_train = datas
    elif len(datas) == 3:
        encoder_input_train, decoder_input_train, decoder_output_train = datas
    
    print('Train data length: ', len(encoder_input_train))
    print('-'*10, 'Load data complete', '-'*10,)  

    #################
    # Load tokenizer
    ################# 
    print('-'*10, 'Load tokenizer', '-'*10,)
    tokenizer = get_tokenizer(args.tokenizer)
    print('-'*10, 'Load tokenizer complete', '-'*10,)
    
    #################
    # Load model
    ################# 
    print('-'*10, 'Load model', '-'*10,)
    model = get_bart_model('gogamza/kobart-summarization', vocab_length= int(len(tokenizer)), tokenizer_name=args.tokenizer)
    print('-'*10, 'Load tokenizer complete', '-'*10,)

    #################
    # Set dataset and trainer
    #################
    print('-'*10, 'Set Dataset and Trainer', '-'*10,)
    dataset, data_collator, eval_dataset = prepare_for_pretraining(tokenizer, encoder_input_train)
    trainer = set_trainer(model, tokenizer, data_collator, dataset, eval_dataset,
                                batch_size=16, accumalation_step=10, epoch=args.epochs)
    print('-'*10, 'Set dataset and trainer complete', '-'*10,)

    #################
    # Start pretraining
    #################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('-'*10, 'Start pretraining:\t', device, '-'*10,)
    trainer.train()
    print('-'*10, 'Pretraing complete', '-'*10,)
    