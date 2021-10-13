#export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
export CLASSPATH=~/Downloads/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

data_path=Samsum/samsum
echo "Start tokenizing.==================================="
# Tokenize hypothesis and target files.
cat ${data_path}/samsum.test.hypo1127 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${data_path}/samsum.test.hypo1127.tokenized
cat ${data_path}/samsum.test.summary | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${data_path}/test.hypo1127.summary

echo "Start rouge scoring.==================================="
files2rouge ${data_path}/samsum.test.hypo1127.tokenized ${data_path}/test.hypo1127.summary
# Expected output: (ROUGE-2 Average_F: 0.21238)