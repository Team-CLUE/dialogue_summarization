#nsml: dacon/nia-tf:1.0

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
    ]
)