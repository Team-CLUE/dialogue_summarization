import os
from glob import glob

import torch
from torch.utils.data import DataLoader

from data.custom_dataset import DatasetForInference
from data.preprocessing import *

import nsml

def bind_model(model, tokenizer, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'model')
        model.save_pretrained(save_dir)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):      
        save_dir = os.path.join(dir_name, 'model/pytorch_model.bin')
        state_dict = torch.load(save_dir) 
        model.load_state_dict(state_dict)
        print("model 로딩 완료!")

    def infer(test_path, **kwparser):
        preprocessor = Preprocess('<usr>', '</s>', 'finetuning')

        test_json_path = os.path.join(test_path, 'test_data', '*')
        print(f'test_json_path :\n{test_json_path}')
        test_path_list = glob(test_json_path)
        test_path_list.sort()
        print(f'test_path_list :\n{test_path_list}')

        test_json_list = preprocessor.make_dataset_list(test_path_list)
        test_data = preprocessor.make_set_as_df(test_json_list)
        test_id = test_data['dialogueID']

        encoder_input_test, decoder_input_test = preprocessor.make_model_input(test_data, is_test= True)
        
        ######################
        encoder_input_test = delete_char(encoder_input_test)

        tokenized_encoder_inputs = tokenizer(list(encoder_input_test), 
                return_tensors="pt", 
                add_special_tokens=True, 
                padding=True, truncation=True, 
                max_length=256, 
                return_token_type_ids=False,)
        #tokenized_decoder_inputs = tokenizer.tokenize(decoder_input_test, return_tensors="pt", padding=True, add_special_tokens=True, truncation=True, max_length=256, return_token_type_ids=False,)
        print(tokenized_encoder_inputs['input_ids'][0:10])

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        dataset = DatasetForInference(tokenized_encoder_inputs, test_id, len(encoder_input_test))
        
        dataloader = DataLoader(dataset, batch_size=128)
        
        summary = []
        text_ids = []
        with torch.no_grad():
            for item in tqdm(dataloader):
                text_ids.extend(item['ID'])
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
                for ids in generated_ids:
                    result = tokenizer.decode(ids, skip_special_tokens=True)
                    index = result.find('.')
                    if index != -1:
                        result = result[:index+1]
                    summary.append(result)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # return list(zip(pred.flatten(), clipped.flatten()))
        
        return list(zip(text_ids, summary))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)