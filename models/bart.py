from transformers import BartConfig, BartForConditionalGeneration

def get_bart_model(
        model_name:str,
        vocab_length:int,
        change_model_size: bool = False,
)->BartForConditionalGeneration:
    '''
        Arguments:
            model_name: str 
                허깅페이스에 있는 pretrained toknizer의 모델 이름
            vocab_length: int
                tokenizer에 정의 되어있는 vocabulary의 개수
            change_model_size: bool
                모델 크기 변경 유무, default = False

        Return
            BartForConditionalGeneration

        Summary:
            허깅페이스에서 Bart config를 로딩하고, BartForConditionalGeneration에 config 및 vocab size를 적용하여 반환
    '''
    config = BartConfig().from_pretrained(model_name)
    if change_model_size:
        config.d_model = 1024
        config.decoder_attention_heads = 16
        config.decoder_ffn_dim = 4096
        config.decoder_layers = 10

        config.encoder_attention_heads = 16
        config.encoder_ffn_dim = 4096
        config.encoder_layers = 10
    model = BartForConditionalGeneration(config=config)
    model.resize_token_embeddings(vocab_length)
    return model
