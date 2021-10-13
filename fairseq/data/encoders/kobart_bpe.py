# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class SentencepieceConfig(FairseqDataclass):
    sentencepiece_model: str = field(
        default="???", metadata={"help": "path to sentencepiece model"}
    )


@register_bpe("kobart", dataclass=SentencepieceConfig)
class SentencepieceBPE(object):
    def __init__(self, cfg):
        sentencepiece_model = file_utils.cached_path(cfg.sentencepiece_model)
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
        except ImportError:
            raise ImportError(
                "Please install kobart with: pip install git+https://github.com/SKT-AI/KoBART#egg=kobart"
            )

    def encode(self, x: str) -> str:
        # return " ".join(self.sp.EncodeAsPieces(x))
        # previous : tokenized text
        # after : changed the format to our dataset: tokenized id
        return x


    def decode(self, x: str) -> str:
        # return x.replace(" ", "").replace("\u2581", " ").strip()
        # final_decode = [self.sp.IdToPiece(int(id)) for id in decode_id.split()]
        # final_decode = ' '.join(map(str, final_decode))
        # final_decode = final_decode.replace('_', '').replace('▁', '')
        decode_id = x.replace("\u2581", " ").strip()
        final_decode = [int(id) for id in decode_id.split()]
        final_decode = self.sp.decode_ids(final_decode)
        return final_decode

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")
