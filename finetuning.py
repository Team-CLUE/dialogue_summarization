import argparse
import nsml
import torch

from data.utill import get_data

from models.toeknizers import get_tokenizer
from models.bart import get_bart_model
from models.train_utill import prepare_for_pretraining, set_trainer

from nsml_setting.nsml import bind_model  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()
    # os.system("wandb login auth코드")

    if args.pause :
        nsml.paused(scope=locals())

    #################
    # Load data
    ################# 
    print('-'*10, 'Load data', '-'*10,)
    datas = get_data('<usr>', '</s>', 'pretraining')
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
    tokenizer = get_tokenizer('gogamza/kobart-summarization')
    print('-'*10, 'Load tokenizer complete', '-'*10,)
    
    #################
    # Load model
    ################# 
    print('-'*10, 'Load model', '-'*10,)
    model = get_bart_model('gogamza/kobart-summarization', len(tokenizer))
    print('-'*10, 'Bind for nsml setting', '-'*10,)
    bind_model(model=model, tokenizer=tokenizer, parser=args)
    if args.load:
        nsml.load(checkpoint=0, session='nia2012/final_dialogue/27')
    print('-'*10, 'Load tokenizer complete', '-'*10,)

    #################
    # Set dataset and trainer
    #################
    print('-'*10, 'Set Dataset and Trainer', '-'*10,)
    


    print('-'*10, 'Set dataset and trainer complete', '-'*10,)

    #################
    # Start pretraining
    #################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('-'*10, 'Start pretraining:\t', device, '-'*10,)
    trainer.train()


    print('-'*10, 'Pretraing complete', '-'*10,)

    nsml.save(0)
    print('-'*10, '저장완료!', '-'*10,)
    