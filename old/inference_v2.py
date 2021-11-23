import argparse
import nsml

import torch
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, BartConfig

from nsml_setting.nsml import bind_model

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
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-summarization')
    special_tokens_dict = {'additional_special_tokens': ['#@URL#','#@이름#','#@계정#','#@신원#','#@전번#',
                '#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#', '#@시스템#사진', '#@시스템#검색',  '#@시스템#지도#', '#@시스템#기타#', '#@시스템#파일#',
                '#@시스템#동영상#', '#@시스템#송금#', '#@시스템#삭제#']}
    tokenizer.add_special_tokens(special_tokens_dict)

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
    generate_model.to('cuda:0')
    
    bind_model(model=generate_model, tokenizer=tokenizer, parser=args)
    nsml.load(checkpoint=9, session='nia2012/final_dialogue/68')
    
    
    if args.pause :
        nsml.paused(scope=locals())

    if args.mode == 'train' :
        nsml.save(0)
            



            