from fairseq.models.lightconv import LightConvModel
# from gluonnlp.data import SentencepieceTokenizer
# from kogpt2.utils import get_tokenizer
import torch


# tok_path = get_tokenizer()
# tokenizer = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)

dynamic = LightConvModel.from_pretrained(
    # model_name_or_path='../raid_model/kodial/200114/',
    model_name_or_path='../model/kodial/custom_token/',
    checkpoint_file='checkpoint_last.pt',
    # checkpoint_file='checkpoint_last.pt',
    data_name_or_path='./kodial-bin', #ln -s ../data/kodial/custom_toked_lang/kodial-bin ../model/kodial/custom_token/kodial-bin
    bpe='sentencepiece',
    sample_break_mode='eos',
    sentencepiece_model='../model/sentencepiece/kodial_sp.spieces.model',
)

dynamic.cuda()
dynamic.eval() # disable dropout
dynamic.half()
count = 1
bsz = 64


while 1:
    print("Please enter [DONE] if you want to exit demo.")
    val = input("Enter your dialogue scripts: ")
    with torch.no_grad():
        output = dynamic.sample(val, beam=4, lenpen=0.9, max_len_b=40, min_len=10, no_repeat_ngram_size=3)
    if val != '[DONE]':
        print(">>>>> Summary: ",output)
        output = " "
    if val == '[DONE]':
        print(">>>>> Bye")
        exit(0)




