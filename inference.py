import torch
from fairseq.models.lightconv import LightConvModel
from setproctitle import setproctitle

setproctitle("Seol")

dynamic = LightConvModel.from_pretrained(
    # 'checkpoints/',
    # 'model/',
    'model/20201127/',
    checkpoint_file='checkpoint_best.pt',
    # data_name_or_path='cnn_dm-bin'
    data_name_or_path='samsum-bin',
    bpe='gpt2',
    sample_break_mode='eou'
)

dynamic.cuda()
dynamic.eval()
dynamic.half()
count = 1
bsz = 32
# with open('cnn_dm/test.source') as source, open('cnn_dm/test.hypo', 'w') as fout:
with open('samsum/samsum.test.dialog') as source, open('samsum/samsum.test.hypo1127', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                # hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
                hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=0.9, max_len_b=200, min_len=35,
                                                  no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        # hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        hypotheses_batch = dynamic.sample(slines, beam=4, lenpen=0.9, max_len_b=200, min_len=35,
                                                  no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()