import sentencepiece as spm
import math
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer

# vocab_file = '/home/seol/kogpt2/pytorch_kogpt2_676e9bcfa7.params'
# model_file = '/home/seol/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
model_file = '../model/sentencepiece/kodial_sp.spieces.model'
sp_model = spm.SentencePieceProcessor()
sp_model.Load(model_file)

tokens = sp_model.encode_as_pieces('떡볶이 먹고 체해서 카스활명수를 마셨지만 효과가 없어서 손을 따야 한다고 말했다.')
ids = sp_model.encode_as_ids('떡볶이 먹고 체해서 카스활명수를 마셨지만 효과가 없어서 손을 따야 한다고 말했다.')

print(tokens)
print(ids)

tokens = sp_model.decode_pieces(tokens)
ids = sp_model.decode_ids(ids)

print(tokens)
print(ids)

tokenizer = SentencepieceTokenizer(model_file, num_best=0, alpha=0)

print(tokenizer('떡볶이 먹고 체해서 카스활명수를 마셨지만 효과가 없어서 손을 따야 한다고 말했다.'))
#=====================================================
# with open('../data/kodial/kogpt2_toked_lang/dict.txt', 'w') as f:
#     for id in range(sp_model.GetPieceSize()):
#         # probability score in unigram language model can be used as freq
#         f.write("%s %s\n" % (sp_model.IdToPiece(id), int(math.exp(sp_model.GetScore(id)))))
# and then remove special tokens from index 1 to 104 (open vi editor > vi dict.txt :1,104d)
#========================================================

# Note(mingdachen): For the purpose of consisent API, we are
# generating a vocabulary for the sentence piece tokenizer.
# vocab = {sp_model.IdToPiece(i): i for i
#               in range(sp_model.GetPieceSize())}
# _token_id = sp_model.piece_to_id(_token)
# print(vocab) # same as encoder.json

# https://github.com/pytorch/fairseq/issues/1186
# https://github.com/google/sentencepiece/issues/248

# result = [[id, sp_model.IdToPiece(id), math.exp(sp_model.GetScore(id))] for id in range(sp_model.GetPieceSize())]
# print(result)
# [[id, sp_model.IdToPiece(id), math.exp(sp_model.GetScore(id))] for id in range(sp_model.GetPieceSize())]




# id = '1681 11217 1042 13916 47437 5 2263 23720 27226 47437 49506 49506 49506 49506 49506 49506 155 134 48075 3337 47780 1343 5 47924 47590 9860 47576 47491 3918 1130 4472 28089 12958 12725 47437 49506 49506 49506 49506 49506 47908 49014 47439 218 9596 8050 28044 36444 49014 47439 302 47638 47450 47578 49108 27078 37246 37246 37246 1130 2602 40468 47466'
# id = '10070 47810 33167 48477 47438 8952 8274 48318 48218 1925 47665 168 48122 47457 10247 48217 31935 10441 155 26086 699 47466 1925 5 3964 25267 47938 3436 47812 206 47454 32952 5524 47661 13916 36076 24517 4880 793 652 47474 47687'
# {'summary': '168 26086 47466 11405 41107 123 47440', 'context': '10070 47810 33167 48477 47438 8952 8274 48318 48218 1925 47665 168 48122 47457 10247 48217 31935 10441 155 26086 699 47466 1925 5 3964 25267 47938 3436 47812 206 47454 32952 5524 47661 13916 36076 24517 4880 793 652 47474 47687'}
# each_id = id.split()
# for id_ in each_id:
#     print(sp_model.IdToPiece(int(id_)))
