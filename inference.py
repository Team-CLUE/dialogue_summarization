
from models.toeknizers import get_tokenizer
from data.utill import get_data
from glob import glob
from transformers import AutoConfig, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "dataset/Validation/*/*"

if __name__ == "__main__":
    
    # set data path
    pathes = glob(DATA_PATH)
    pathes.sort()

    # get data
    encoder_input_test, decoder_input_test, ground_trues =\
         get_data('<usr>', '</s>', 'finetuning', pathes)
    
    print(f"Test data length : {len(encoder_input_test)}")

    # get tokenizer
    tokenizer = get_tokenizer('gogamza/kobart-summarization')

    # Load model
    model_path = "모델 bin file과 config file이 있는 directory"
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=config)

    # Dataloader

    # Start Inference
    with torch.no_grad():
        for 
