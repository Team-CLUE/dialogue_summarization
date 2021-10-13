import math
import os
from subprocess import check_output
import numpy as np
import sentencepiece as spm
import math
# from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

# root_path = '../model/sentencepiece/'
# vocab_name = 'kodial_sp.spieces.vocab'
# vocab_path = str(os.path.join(root_path, vocab_name))
# output_path = '../data/kodial/custom_toked_lang/dict.txt'

root_path = '../model/sentencepiece_v2/'
vocab_name = 'kodial_sp_v2.spieces.vocab'
vocab_path = str(os.path.join(root_path, vocab_name))
output_path = '../data/kodial_v2_shuffle/custom_toked_lang/dict.txt'


p_ = check_output(['wc', '-l', vocab_path])

print("============Original vocabulary size is {}.....".format(p_))

count = 0
with open(vocab_path, 'r', encoding='utf-8') as fr, open(output_path, 'w') as fw:
    for line in fr:
        count += 1
        if count > 3: # also need to remove <unk>, <s> and </s> from the sentencepiece dictionary with a dummy count of 100
            word = line.split('\t')[0]
            fw.write("%s %s\n" % (word, int(100)))

p = check_output(['wc', '-l', output_path])

print("============New vocabulary size is {}.....this vocab size should be {}".format(p, np.subtract(int(p_.split()[0]), 3)))

# ================================================================ kogpt2 dictionary build

# vocab_file = '/home/seol/kogpt2/pytorch_kogpt2_676e9bcfa7.params'
# model_file = '/home/seol/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
# sp_model = spm.SentencePieceProcessor()
# sp_model.Load(model_file)
#
# with open('../data/kodial/kogpt2_toked_lang/dict.txt', 'w') as f:
#     for id in range(sp_model.GetPieceSize()):
#         # probability score in unigram language model can be used as freq
#         f.write("%s %s\n" % (sp_model.IdToPiece(id), int(math.exp(sp_model.GetScore(id)))))
# # and then remove special tokens from index 1 to 104 (open vi editor > vi dict.txt :1,104d)

# ================================================================ kogpt2 dictionary build for changing freq
# root_path = '../data/kodial/kogpt2_toked_lang/'
# vocab_name = 'dict.txt'
# vocab_path = str(os.path.join(root_path, vocab_name))
# output_path = '../data/kodial/kogpt2_toked_lang/dict_.txt'
#
# p_ = check_output(['wc', '-l', vocab_path])
#
# print("============Original vocabulary size is {}.....".format(p_))
#
# count = 0
# with open(vocab_path, 'r', encoding='utf-8') as fr, open(output_path, 'w') as fw:
#     for line in fr:
#         count += 1
#         # if count > 3: # also need to remove <unk>, <s> and </s> from the sentencepiece dictionary with a dummy count of 100
#         #     word = line.split('\t')[0]
#         #     fw.write("%s %s\n" % (word, int(100)))
#         if count > 104: # also need to remove <unk>, <s> and </s> from the sentencepiece dictionary with a dummy count of 100
#             word = line.split(' ')[0]
#             fw.write("%s %s\n" % (word, int(100)))
#
# p = check_output(['wc', '-l', output_path])
#
# print("============New vocabulary size is {}.....this vocab size should be {}".format(p, np.subtract(int(p_.split()[0]), 3)))