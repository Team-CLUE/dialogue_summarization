from fairseq.models.lightconv import LightConvModel
import torch
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import logging

logging.getLogger("flask_ask").setLevel(logging.DEBUG)

STATUS_OK = "ok"
STATUS_ERROR = "error"

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


dynamic = LightConvModel.from_pretrained(
    model_name_or_path='../model/kodial/custom_token/',
    checkpoint_file='checkpoint_last.pt',
    data_name_or_path='./kodial-bin', #ln -s ../data/kodial/custom_toked_lang/kodial-bin ../model/kodial/custom_token/kodial-bin
    bpe='sentencepiece',
    sample_break_mode='eos',
    sentencepiece_model='../model/sentencepiece/kodial_sp.spieces.model',
)

dynamic.cuda()
dynamic.eval() # disable dropout
dynamic.half()


def run_model(input_text):

    logger.info("Running the dialog summarization model ...")

    while 1:
        prompt = str(input_text)
        with torch.no_grad():
            output = dynamic.sample(prompt, beam=4, lenpen=0.9, max_len_b=40, min_len=10, no_repeat_ngram_size=3)
        return output


@app.route('/', methods=['GET'])
def api_input():
    if 'input' in request.args:
        input = str(request.args['input'])
    else:
        return "Error: No input field provided. Please insert an input."

    results = run_model(input)

    return jsonify(results)

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=45100, debug=True, use_reloader=False)