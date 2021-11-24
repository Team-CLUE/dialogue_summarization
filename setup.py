#nsml:scatterlab/python-mecab-ko:3.7-circleci-node

from distutils.core import setup
setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'tokenizers==0.10.3',
        'transformers==4.11.3',
        'torch==1.7.1',
        'soynlp',
        'rouge',
        'konlpy',
        'python-dev-tools',
        'python-mecab-ko',
        'tweepy==3.7'

    ]
)