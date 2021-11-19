
from models.toeknizers import get_tokenizer
from data.utill import get_data
from glob import glob
from transformers import AutoConfig, BartForConditionalGeneration
import torch
import numpy as np
import rouge
from tqdm import tqdm
from torch.utils.data import DataLoader
from train.train_utill import prepare_for_inference

DATA_PATH = "dataset/Validation/*/*"

if __name__ == "__main__":
    
    # set data path
    pathes = glob(DATA_PATH)
    pathes.sort()

    # get data
    encoder_input_test, _ , ground_trues =\
         get_data('<usr>', '</s>', 'finetuning', pathes)
    
    print(f"Test data length : {len(encoder_input_test)}")

    # get tokenizer
    tokenizer = get_tokenizer('gogamza/kobart-summarization')

    # Load model
    model_path = "pre-trained/all_data/checkpoint-90000"
    config = AutoConfig.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
    
    # Dataloader
    inference_dataloader =\
         prepare_for_inference(tokenizer, encoder_input_test, batch_size=16)
    
    # Start Inference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        model.eval()
        model.to(device)
        
        predictions = np.array([])

        for inputs_ids, _ in tqdm(inference_dataloader):
            
            inputs_ids = inputs_ids.to(device)

            generated_ids = model.generate(input_ids=inputs_ids, 
                                # do_sample=True, 
                                # max_length=50, 
                                # top_p=0.92, #92%로 설정하고 샘플링하기
                                # top_k=0
                                no_repeat_ngram_size=2, 
                                early_stopping=True,
                                max_length=50, 
                                num_beams=5,
                            )
            predicted_s = np.array(list(map(tokenizer.decode, generated_ids)))
            predictions = np.append(predictions, predicted_s)
    
    