#nsml: dacon/nia-tf:1.0

from distutils.core import setup

import tokenizers

setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'tokenizers',
        'transformers'
    ]
)