import argparse
import re
import sentencepiece as spm

class PreprocessForRouge():
    REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9가-힣]")

    def __init__(self):
        self.model_file = '../model/sentencepiece_v2/kodial_sp_v2.spieces.model'

    def convert_into_id(self, text):
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(self.model_file)
        ids = sp_model.encode_as_ids(text)
        return ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help='input file path')
    parser.add_argument('--outfile', help='output file path')
    args = parser.parse_args()

    root = PreprocessForRouge()

    with open(args.infile, 'r', encoding='utf-8') as fin, open(args.outfile, 'w', encoding='utf-8') as fout:
        for line in fin:
            sentence = root.REMOVE_CHAR_PATTERN.sub(" ", line.lower()).strip()
            # tokens = root.tokenize_text(root.REMOVE_CHAR_PATTERN.sub(" ", sentence))
            # toked_sentence = " ".join(tokens)
            # fout.write("%s\n" % (toked_sentence))

            ids = root.convert_into_id(str(sentence))
            toked_sentence = " ".join(map(str, ids))
            fout.write("%s\n" % (toked_sentence))


