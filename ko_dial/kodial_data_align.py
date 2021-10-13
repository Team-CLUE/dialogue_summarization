from torch.utils.data import DataLoader
from tqdm import tqdm
import jsonlines
# from sklearn.model_selection import train_test_split
from torch.utils.data import RandomSampler

'''
please do pip install datasets
Description
Reddit dataset, where TIFU denotes the name of subbreddit /r/tifu. As defined in the publication, styel "short" uses title as summary and "long" uses tldr as summary.

Features includes:

documents: post text without tldr.
tldr: tldr line.
title: trimmed title without tldr.
ups: upvotes.
score: score.
num_comments: number of comments.
upvote_ratio: upvote ratio.

'''

# TOKEN='<eou>'
BOS = '<s>'
EOS = '</s>'

class Dataset(object):
    def __init__(self, data):
        self.samples = self.create_samples(data)

    def __getitem__(self, idx):
        return {
            'topic': self.samples[idx]['topic'],
            'conv_id': self.samples[idx]['conv_id'],
            'summary': self.samples[idx]['summary'],
            'context': self.samples[idx]['context'],
        }

    def __len__(self):
        return len(self.samples)

    def create_samples(self, data):
        samples = []

        for sample in tqdm(data):
            # print(train['documents'][0].replace('\n', ''))
            # summary = sample['요약']
            summary = sample['body']['summary']
            summary = summary.lstrip().replace('\n\n', '').replace('\r\n', '').replace('\n', '').replace('\r', '').replace('\t', '')

            context = ["".join(line['utterance'].lstrip().replace('\n', '').replace('\r', '').replace('\t', '')) for line in sample['body']['dialogue']]
            context = EOS.join(BOS + str(u) for u in context) + EOS # <s>멋지네요 언니</s> 이거는 모델 학습시에 넣는걸로

            samples.append({
                'topic': sample['header']['dialogueInfo']['topic'],
                'conv_id': sample['header']['dialogueInfo']['dialogueID'],  # str
                'summary': summary,
                'context': context, })

        return samples

    def collate_fn(self, batch):
        topic = [sample['topic'] for sample in batch]
        summary = [sample['summary'] for sample in batch]
        context = [sample['context'] for sample in batch]
        batch = {
            'topic': [sample['topic'] for sample in batch],
            'conv_id': [sample['conv_id'] for sample in batch],
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
        topic = line['topic']
        id = line['conv_id']
        summary = line['summary']
        dialog = line['context']

        for t, i, s, d in zip(topic, id, summary, dialog):
            batch = {
                'topic': t,
                'conv_id': i,
                'summary': s,
                'context': d,
            }
            file_index.write("%s\n" % batch)
        # file_index.write("%s\n" % line)
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
            datas.append(batch)
        # else:
        #     if count == 1:
        #         tqdm.write("-------Build original {} file...".format(file_type))
        #     datas.append(batch)
    # if type == 'align':
    #     return data_src, data_tgt
    # else:
    #     return datas
    return data_src, data_tgt, datas

def main(input_path, output_path_src, output_path_tgt, output_path_ori, file_type):

    dataloader = Dataset(input_path)
    sampler = RandomSampler(dataloader)
    dataloader_ = DataLoader(
        dataset=dataloader,
        sampler=sampler,  # including shuffle
        collate_fn=dataloader.collate_fn,
        batch_size=16, )

    # file_type = str(input_path.split('/')[1].split('.')[0])
    # datas = build_data(dataloader_, type='original', file_type=file_type)
    data_src, data_tgt, datas = build_data(dataloader_, type='align', file_type=file_type)

    print("len(context):", len(data_src), "len(summary):", len(data_tgt), "len(total):", len(datas))

    build_file(data_src, data_tgt, datas,
               output_path_src, output_path_tgt, output_path_ori)

if __name__=='__main__':

    # dataset = load_dataset("reddit_tifu", 'long', split='train')  # args2 = 'long' or 'short'
    # dataset = load_dataset('json', data_files='../data/kodial/orig/data.jsonl')
    #
    # train_test = dataset.train_test_split(test_size=0.1, seed=42)
    # train_test['train'], train_test['validation'] = train_test['train'].train_test_split(test_size=0.1, seed=42).values()

    # with jsonlines.open('../data/kodial/orig/data.jsonl') as reader:
    with jsonlines.open('../data/kodial_v2/split_result/ko_conv_summary_data.train.jsonl') as reader:
        train = [obj for obj in reader]

    with jsonlines.open('../data/kodial_v2/split_result/ko_conv_summary_data.valid.jsonl') as reader:
        valid = [obj for obj in reader]

    with jsonlines.open('../data/kodial_v2/split_result/ko_conv_summary_data.test.jsonl') as reader:
        test = [obj for obj in reader]

    # to_train, test = train_test_split(data, test_size = 0.1, random_state=21)
    # train, val = train_test_split(to_train, test_size = 0.1)

    # print(train_test)
    # input_train_path = train_test['train']
    # input_val_path = train_test['validation']
    # input_test_path = train_test['test']

    input_train_path = train
    input_val_path = valid
    input_test_path = test

    prefix = 'kodial'
    output_root_path = '../data/kodial_v2_shuffle/'
    source = '.dialog'
    target = '.summary'
    ori = '.original'

    output_train_path_src = output_root_path + prefix + '.train' + source
    output_train_path_tgt = output_root_path + prefix + '.train' + target
    output_train_path_ori = output_root_path + prefix + '.train' + ori

    output_val_path_src = output_root_path + prefix + '.validation' + source
    output_val_path_tgt = output_root_path + prefix + '.validation' + target
    output_val_path_ori = output_root_path + prefix + '.validation' + ori

    output_test_path_src = output_root_path + prefix + '.test' + source
    output_test_path_tgt = output_root_path + prefix + '.test' + target
    output_test_path_ori = output_root_path + prefix + '.test' + ori

    main(input_train_path, output_train_path_src, output_train_path_tgt, output_train_path_ori, 'train')
    main(input_val_path, output_val_path_src, output_val_path_tgt, output_val_path_ori, 'validation')
    main(input_test_path, output_test_path_src, output_test_path_tgt, output_test_path_ori, 'test')