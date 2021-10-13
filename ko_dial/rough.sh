output_path=../result
data_path=../data/kodial_v2_shuffle

echo "Start tokenization.==================================="
# --infile "path/to/input/file" --outfile "path/to/output/file"

# inference result tokenization
#python kodial_rouge.py --infile ../result/kodial.test.checkpoint_best.0225 --outfile ../result/kodial.test.checkpoint_best.0225.tokenized
# evaluation dataset tokenization
#python kodial_rouge.py --infile ${data_path}/kodial.test.summary --outfile ../result/kodial_v2_shuffle.test.summary.tokenized

echo "Start ROUGE scoring.==================================="
# files2rouge {inference file} {test set file}
files2rouge ${output_path}/kodial.test.checkpoint_best.0225.tokenized ../result/kodial_v2_shuffle.test.summary.tokenized

