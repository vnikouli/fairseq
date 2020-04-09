# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    LRUCacheDataset
)
from fairseq.tasks import register_task
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
from fairseq.tasks.masked_lm import MaskedLMTask

logger = logging.getLogger(__name__)


@register_task('masked_tok_pos_lm')
class MaskedLMTokPosTask(MaskedLMTask):
    """Task for training masked language models (e.g., BERT, RoBERTa).
    Mask LM with auxillary sequence labels
    """
    @staticmethod
    def add_args(parser):
        MaskedLMTask.add_args(parser)
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data-aux', help='colon separated path to additional data directories list')

    def __init__(self, args, dictionary, dictionary_aux):
        super().__init__(args, dictionary)
        self.dictionary_aux = dictionary_aux
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        paths_aux = utils.split_paths(args.data_aux)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        dictionary_aux = Dictionary.load(os.path.join(paths_aux[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('dictionary sup: {} types'.format(len(dictionary_aux)))
        return cls(args, dictionary, dictionary_aux)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        paths_aux = utils.split_paths(self.args.data_aux)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        data_aux_path = paths_aux[(epoch - 1) % len(paths_aux)]
        split_aux_path = os.path.join(data_aux_path, split)


        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        aux_dataset = data_utils.load_indexed_dataset(
            split_aux_path,
            self.source_dictionary_aux,
            self.args.dataset_impl,
            combine=combine
        )
        if aux_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_aux_path))
        import pdb

        # create continuous blocks of tokens
        aux_dataset = TokenBlockDataset(
            aux_dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary_aux.pad(),            
            eos=self.source_dictionary_aux.eos(),
            break_mode=self.args.sample_break_mode
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),            
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode
        )
        #pdb.set_trace()


        logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        aux_dataset = PrependTokenDataset(aux_dataset, self.source_dictionary_aux.bos())
        
        # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args.mask_whole_words else None

        src_dataset, tgt_dataset, tgt_dataset_aux = MaskTokensDataset.apply_mask_mt(
            dataset,
            aux_dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )


        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(

            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'target': PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'aux_target': PadDataset(
                        aux_dataset,
                        pad_idx=self.source_dictionary_aux.pad(),
                        left_pad=False,
                    ),
                    


                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },





                sizes=[src_dataset.sizes],


            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )


    @property
    def source_dictionary_aux(self):
        return self.dictionary_aux

