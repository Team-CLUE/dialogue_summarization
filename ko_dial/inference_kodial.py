import torch
from fairseq.models.lightconv import LightConvModel
from setproctitle import setproctitle
import re
from gluonnlp.data import SentencepieceTokenizer
# from kogpt2.utils import get_tokenizer

setproctitle("Seol")

# tok_path = get_tokenizer()
# tokenizer = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)

dynamic = LightConvModel.from_pretrained(
    # model_name_or_path='../raid_model/kodial/200114/',
    # model_name_or_path='../model/kodial/custom_token/',
    # model_name_or_path='../model/kodial/kogpt2_token/',
    model_name_or_path='../model/kodial_v2_shuffle/custom_token/',
    checkpoint_file='checkpoint_last.pt',
    # checkpoint_file='checkpoint_last.pt',
    data_name_or_path='./kodial-bin', #ln -s ../data/kodial/custom_toked_lang/kodial-bin ../model/kodial/custom_token/kodial-bin
    bpe='sentencepiece',
    # bpe='gpt2',
    sample_break_mode='eos',
    sentencepiece_model='../model/sentencepiece_v2/kodial_sp_v2.spieces.model',
    # sentencepiece_model='../model/sentencepiece/kodial_sp.spieces.model',
)

dynamic.cuda()
dynamic.eval() # disable dropout
dynamic.half()
count = 1
bsz = 256
# with open('cnn_dm/test.source') as source, open('cnn_dm/test.hypo', 'w') as fout:
# with open('samsum/samsum.test.dialog') as source, open('samsum/samsum.test.hypo1127', 'w') as fout:
# with open('../data/kodial/kodial.train.dialog', encoding='utf-8') as source, open('../result/kodial.train.checkpoint_best.0120', 'w', encoding='utf-8') as fout:
with open('../data/kodial_v2_shuffle/kodial.test.dialog', encoding='utf-8') as source, open('../result/kodial.test.checkpoint_last.0225', 'w', encoding='utf-8') as fout:
    sline = source.readline().strip()
    slines = re.findall('<s>(.+?)</s>', sline)
    # slines = [sline]
    slines_ = ' '.join(map(str, slines))
    slines = [slines_]
    # print(sliness)
    # exit(0)
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=0.9, max_len_b=40, min_len=10,
                                                  no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1


    if slines != []:
        hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=0.9, max_len_b=40, min_len=10,
                                                  no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()


