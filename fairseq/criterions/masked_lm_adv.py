# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_lm_adv')
class MaskedLmAdvLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    this one combines token and Pos adversarial prediction : we want to be able to predict token correctly but not it's PoS
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)


        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens.fill_(True)
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        model_out = model(**sample['net_input'], masked_tokens=masked_tokens)[0]

        logits=model_out[0]
        logits_aux =model_out[1]
        targets, aux_targets, ratio = model.get_targets(sample, [logits])
        targets = targets[masked_tokens]
        aux_targets = aux_targets[masked_tokens]

        primary_loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        aux_loss = modules.cross_entropy(
            logits_aux.view(-1, logits_aux.size(-1)),
            aux_targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # minimize primary loss and maximize auxilary loss
        alpha=ratio

            
        sample_size = masked_tokens.int().sum()
        logging_output = {
            'loss': primary_loss.data,
            'aux_loss': aux_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }

        return (primary_loss, aux_loss), sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        aux_loss_sum = sum(log.get('aux_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('aux_loss', aux_loss_sum / sample_size / math.log(2), sample_size, round=3)

        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
