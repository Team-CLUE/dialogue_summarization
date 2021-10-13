import argparse
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import os
import jsonlines
from tqdm import tqdm
from torch.utils.data import DataLoader

class Dataset(object):
    def __init__(self, data_):
        self.samples = self.create_samples(data_)

    def __getitem__(self, idx):
        return {
            'summary': self.samples[idx]['summary'],
            'context': self.samples[idx]['context'],
        }

    def __len__(self):
        return len(self.samples)

    def create_samples(self, data_):
        # json.loads(open(json_path).read())
        samples = []

        for sample in tqdm(data_):
            # print(train['documents'][0].replace('\n', ''))
            sample = sample['body']
            summary = sample['summary']
            summary = summary.lstrip().replace('\n\n', '').replace('\r\n', '')\
                .replace('\n', '').replace('\r','').replace('\t', '')
            context = ["".join(line['utterance'].lstrip().replace('\n', '').replace('\r', '').replace('\t', '')) for
                       line in sample['dialogue']]

            samples.append({
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

def build_data(input_data):
    datas = []
    count = 0
    for batch in tqdm(input_data):
        count += 1
        if count == 1:
            tqdm.write("-------Building for sentencepiece file...")
        datas.append(batch)

    return datas

def build_file(input_src, file_name_src):
    ''' source : context,
        target : summary.
        total : context + summary '''

    file_src = open(file_name_src, 'w')

    for line in input_src:
        file_src.write("%s\n" % '\n'.join(map(str, line['summary'])))
        # utter = '\n'.join(map(str, line['context']))
        for each in line['context']:
            for utter in each:
                file_src.write("%s\n" % utter)

    file_src.close()

def main(input_d):

    dataloader = Dataset(input_d)
    dataloader_ = DataLoader(
        dataset=dataloader,
        collate_fn=dataloader.collate_fn,
        batch_size=16, )

    datas = build_data(dataloader_)

    # for line in datas:
    #     print(line)
    #     break
    # print("len(context):", len(data_src), "len(summary):", len(data_tgt), "len(total):", len(datas))

    # build_file(datas, '../data/kodial/orig/kodial_sp_train.txt')
    build_file(datas, '../data/kodial_v2/kodial_sp_train_v2.txt')

class SPTrain():
    def __init__(self, model_name):
        self.input_file = '../data/kodial_v2/kodial_sp_train_v2.txt'
        self.vocab_size = 32000

        # self.sp_model_root='../model/sentencepiece'
        self.sp_model_root = '../model/sentencepiece_v2'
        self.model_type = 'unigram'  # 학습할 모델 선택, unigram이 더 성능이 좋음'bpe'
        self.character_coverage = 1.0  # 전체를 cover 하기 위해, default=0.9995
        self.sp_model_name = model_name

    def sp_train(self):
        if not os.path.isdir(self.sp_model_root):
            os.mkdir(self.sp_model_root)
        # sp_model_name = 'tokenizer_%d' % (self.vocab_size)
        sp_model_path = os.path.join(self.sp_model_root, self.sp_model_name)

        # user_defined_symbols = ''
        input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s'
        cmd = input_argument % (
        self.input_file, sp_model_path, self.vocab_size, self.model_type, self.character_coverage)

        spm.SentencePieceTrainer.Train(cmd)
        print('sentencepiece train done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_build', help='Data builder for sentencepiece training')
    parser.add_argument('--train', help='sentencepiece training')
    args = parser.parse_args()

    if args.data_build == 'true':
        # with jsonlines.open('../data/kodial/orig/data.jsonl') as reader:
        with jsonlines.open('../data/kodial_v2/split_result/ko_conv_summary_data.all.jsonl') as reader:
            data_ = [obj for obj in reader]

        main(data_)

    if args.train == 'true':
        sp = SPTrain('kodial_sp_v2.spieces')
        sp.sp_train()