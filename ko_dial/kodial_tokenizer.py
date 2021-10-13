import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from gluonnlp.data import SentencepieceTokenizer
from demjson import decode
import torch
import re
import itertools
import argparse
# import sentencepiece as spm

'''
please do pip install datasets
Description

Features includes:
'''

class Encoder(object):
    def __init__(self, args):
        self.which_tokenizer = args.tokenizer
        self.tokenizer = self.load_tokenizer()
        # self.tokenizer.add_tokens([self.args.eou_token])
        # self.model.resize_token_embeddings(len(self.tokenizer))

    def load_model(self):
        if self.which_tokenizer == 'kogpt2':
            from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

            model, vocab = get_pytorch_kogpt2_model()
        elif self.which_tokenizer == 'kobart':
            from kobart import get_pytorch_kobart_model
            '''Todo: model, vocab process 
            reference kobart/infer.py code///// uncompleted'''
            model = None
            inputs = self.tokenizer([], return_tensors='pt')
            vocab = inputs['input_ids']


        return model, vocab

    def sp_tokenizer(self):
        return

    def load_tokenizer(self):
        if self.which_tokenizer == 'kogpt2':
            from kogpt2.utils import get_tokenizer

            tok_path = get_tokenizer()
            tokenizer = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
        elif self.which_tokenizer == 'kobart':
            from kobart import get_kobart_tokenizer

            tokenizer = get_kobart_tokenizer()

        elif self.which_tokenizer == 'custom':
            model_root = '../model/sentencepiece_v2/'
            model_name = 'kodial_sp_v2.spieces.model'
            tok_path = os.path.join(model_root, model_name)

            tokenizer = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
        else:
            print("Please see the parameter option")
        return tokenizer

    def encode(self, sentence):
        toked = self.tokenizer(sentence)

        return toked


class Dataset(object):
    def __init__(self, args, data):
        self.encoder = Encoder(args)

        if args.tokenizer == 'kogpt2':
            self.model, self.vocab = self.encoder.load_model()
        self.samples = self.create_samples(data)


    def __getitem__(self, idx):
        return {
            # 'id': self.samples[idx]['id'],
            'summary': self.samples[idx]['summary'],
            'context': self.samples[idx]['context'],
        }

    def __len__(self):
        return len(self.samples)

    def convert_into_toked(self, sentence):
        return self.encoder.encode(sentence)

    def convert_into_ids(self, tokenized):
        # print(self.vocab)
        # ids = torch.tensor([self.vocab[self.vocab.bos_token], ] + self.vocab[tokenized]).unsqueeze(0)
        ids = torch.tensor(self.vocab[tokenized]).unsqueeze(0)[0].numpy()
        return ids

    def create_samples(self, data):
        samples = []
        for sample in tqdm(data):
            sample = decode(sample)

            summary = self.convert_into_toked(sample['summary']) # toked word
            # summary = self.convert_into_ids(summary) # token id

            utter_match = re.findall('<s>(.+?)</s>', sample['context'])
            utterances = utter_match
            context = [self.convert_into_toked(utter) for utter in utterances]
            # utter_id = list(itertools.chain(*[self.convert_into_ids(each) for each in context])) # token id
            utter_id = list(itertools.chain(*[each for each in context])) # toked word

            context = ' '.join(map(str, utter_id)) # format to string
            summary = ' '.join(map(str, summary))

            samples.append({
                # 'id': sample['id'],  # str
                'summary': summary,
                'context': context, })

        return samples

    def collate_fn(self, batch):
        summary = [sample['summary'] for sample in batch]
        context = [sample['context'] for sample in batch]
        batch = {
            'summary': summary,
            'context': context, }
        return batch

def build_file(input_src, input_tgt, input_total,
               file_name_src, file_name_tgt, file_name_index):
    ''' source : context,
        target : summary.
        total : context + summary '''

    file_src = open(file_name_src, 'w')
    file_tgt = open(file_name_tgt, 'w')
    file_index = open(file_name_index, 'w')

    for line in input_src:
        for sam in line:
            file_src.write("%s\n" % sam)
    file_src.close()

    for line_ in input_tgt:
        for sam in line_:
            file_tgt.write("%s\n" % sam)
    file_tgt.close()

    for line in input_total:
        file_index.write("%s\n" % line)
    file_index.close()

def build_data(input_data, type: str, file_type: str):
    datas, data_src, data_tgt = [], [], []
    count = 0
    for batch in tqdm(input_data):
        count += 1

        if type == 'align':
            if count == 1:
                tqdm.write("-------Build aligned {} file...".format(file_type))
            data_src.append(batch['context'])
            data_tgt.append(batch['summary'])
        else:
            if count == 1:
                tqdm.write("-------Build original {} file...".format(file_type))
            datas.append(batch)
    if type == 'align':
        return data_src, data_tgt
    else:
        return datas

def main(args, input_path, output_path_src, output_path_tgt, output_path_ori, file_type):

    dataloader = Dataset(args, input_path)
    dataloader_ = DataLoader(
        dataset=dataloader,
        collate_fn=dataloader.collate_fn,
        batch_size=16, )

    # file_type = str(input_path.split('/')[1].split('.')[0])
    datas = build_data(dataloader, type='original', file_type=file_type)
    data_src, data_tgt = build_data(dataloader_, type='align', file_type=file_type)

    print("len(context):", len(data_src), "len(summary):", len(data_tgt), "len(total):", len(datas))

    build_file(data_src, data_tgt, datas,
               output_path_src, output_path_tgt, output_path_ori)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', help='Tokenizer selection: kobart or kogpt2 or custom')
    args = parser.parse_args()

    input_root_path = '../data/kodial_v2_shuffle/'
    prefix = 'kodial'
    output_root_path = '../data/kodial_v2_shuffle/custom_toked_lang/'
    source = '.dialog'
    target = '.summary'
    ori = '.original'

    with open(input_root_path + prefix + '.train' + ori) as rt:
        train_data = [obj for obj in rt]

    with open(input_root_path + prefix + '.validation' + ori) as rv:
        val_data = [obj for obj in rv]

    with open(input_root_path + prefix + '.test' + ori) as rtt:
        test_data = [obj for obj in rtt]

    input_train_path = train_data
    input_val_path = val_data
    input_test_path = test_data

    output_train_path_src = output_root_path + prefix + '.train' + source
    output_train_path_tgt = output_root_path + prefix + '.train' + target
    output_train_path_ori = output_root_path + prefix + '.train' + ori

    output_val_path_src = output_root_path + prefix + '.validation' + source
    output_val_path_tgt = output_root_path + prefix + '.validation' + target
    output_val_path_ori = output_root_path + prefix + '.validation' + ori

    output_test_path_src = output_root_path + prefix + '.test' + source
    output_test_path_tgt = output_root_path + prefix + '.test' + target
    output_test_path_ori = output_root_path + prefix + '.test' + ori

    main(args, input_train_path, output_train_path_src, output_train_path_tgt, output_train_path_ori, 'train')
    main(args, input_val_path, output_val_path_src, output_val_path_tgt, output_val_path_ori, 'validation')
    main(args, input_test_path, output_test_path_src, output_test_path_tgt, output_test_path_ori, 'test')