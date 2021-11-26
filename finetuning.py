import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch

from data.utill import get_data

from models.toeknizers import get_tokenizer
from models.bart import get_bart_model
from train.train_utill import prepare_for_finetuning, set_trainer_for_finetuning
#from train.fine_train import finetuning

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
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
    datas = get_data(decoder_strat_token, end_of_sentence, 'finetuning')
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
    if args.load is not None:
        model = get_bart_model(args.load, vocab_length= int(len(tokenizer)), tokenizer_name=args.tokenizer)
        save_dir = os.path.join(args.load, 'pytorch_model.bin')
        state_dict = torch.load(save_dir) 
        model.load_state_dict(state_dict)
    else:    
        model = get_bart_model('gogamza/kobart-summarization', vocab_length= int(len(tokenizer)), tokenizer_name=args.tokenizer)
    print('-'*10, 'Load tokenizer complete', '-'*10,)

    #################
    # Set dataset and trainer
    #################
    print('-'*10, 'Set Dataset and Trainer', '-'*10,)
    train_dataset, valid_dataset = prepare_for_finetuning(tokenizer, encoder_input_train, decoder_input_train, decoder_output_train, batch_size=16)
    trainer = set_trainer_for_finetuning(model, tokenizer, train_dataset, valid_dataset, epoch=args.epochs, batch_size=16, accumalation_step=10)
    print('-'*10, 'Set dataset and trainer complete', '-'*10,)

    #################
    # Start finetuning
    #################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('-'*10, 'Start finetuning:\t', device, '-'*10,)
    trainer.train()
    #finetuning(model.to(device), train_loader, valid_loader, epochs = args.epochs, accumalation_step = 10)
    print('-'*10, 'finetuning complete', '-'*10,)
    