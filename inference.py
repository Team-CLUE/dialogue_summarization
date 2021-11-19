
from models.toeknizers import get_tokenizer
from data.utill import get_data
from glob import glob
from transformers import AutoConfig, AutoModel
import torch
import numpy as np
import rouge
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
    model_path = "모델 bin file과 config file이 있는 directory"
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=config)

    # Dataloader
    inference_dataloader = prepare_for_inference(tokenizer, encoder_input_test, batch_size=16)
    
    # Start Inference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        model.eval()
        model.to(device)
        
        predictions = np.array()

        for inputs in inference_dataloader:
            predicted_s = model(inputs.to(device)).detach().cpu().numpy()
            predictions = np.append(predictions, predicted_s)
    
    