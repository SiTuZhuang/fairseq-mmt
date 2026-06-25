# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_with_mmconsis')
class LabelSmoothedCrossEntropyCriterionWithMMConsis(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        loss, nll_loss, consis_loss_val, es_mean, es_std, contrast_loss_val, entity_push_val = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'consis_loss': float(consis_loss_val) if hasattr(consis_loss_val,'item') else consis_loss_val,
            'entity_score_mean': float(es_mean) if hasattr(es_mean,'item') else es_mean,
            'entity_score_std': float(es_std) if hasattr(es_std,'item') else es_std,
            'contrast_loss': float(contrast_loss_val) if hasattr(contrast_loss_val,'item') else contrast_loss_val,
            
            
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        txt_out = net_output[1]['txt_out']
        img_out = net_output[1]['img_out']
        entity_score = net_output[1].get('entity_score', None)

        txt_pooled = utils.meanpooling_tensor(txt_out)
        img_pooled = utils.meanpooling_tensor(img_out)

        # Forward KL: keep text close to image (suppress inconsistency)
        kl_forward = utils.multimodel_consis_loss(txt_pooled, img_pooled)

        if entity_score is not None:
            # Bidirectional adaptive entity distillation
            entity_t = entity_score.transpose(0, 1).squeeze(-1)   # (B, T)
            entity_w = entity_t.mean(dim=-1)                      # (B,)

            # Reverse KL: image teaches text (entity-like tokens only)
            kl_reverse = utils.multimodel_consis_loss(img_pooled.detach(), txt_pooled)

            w_fwd = 1.0 - entity_w
            w_rev = entity_w
            consis_loss = (w_fwd * kl_forward + w_rev * kl_reverse).mean()
            # V10: Repel entity_score from 0.5 saddle point (entropy traps at 0.5)
            entity_push = -((entity_score - 0.5) ** 2).mean()
            consis_loss = consis_loss + 0.1 * entity_push

            # === Phase3: Entity-weighted Cross-modal Contrastive Distillation ===
            if txt_out is not None:
                B = txt_out.shape[1]  # Phase3/4: batch size for contrastive
                # Sentence-level entity weight (robust to T mismatch)
                # Use sentence-pooled representations directly
                ew = entity_score.float().mean()  # scalar, just indicate entity_score is alive
                txt_ent = txt_pooled
                img_ent = img_pooled
                txt_ent = F.normalize(txt_ent, dim=-1)
                img_ent = F.normalize(img_ent, dim=-1)
                tau = 0.07
                logits = txt_ent @ img_ent.T / tau
                labels = torch.arange(B, device=logits.device)
                loss_t2i = F.cross_entropy(logits, labels)
                loss_i2t = F.cross_entropy(logits.T, labels)
                contrast_loss = (loss_t2i + loss_i2t) / 2

                # === Phase4: Dynamic contrast weight (warmup 0.02->0.1 over 5k steps) ===
                try:
                    nu = model.num_updates
                except:
                    nu = 0
                contrast_coeff = 0.02 + 0.08 * min(float(nu) / 5000.0, 1.0)
                consis_loss = consis_loss + contrast_coeff * contrast_loss
            else:
                contrast_loss = torch.tensor(0.0)
        else:
            consis_loss = kl_forward
            contrast_loss = torch.tensor(0.0)

        consis_loss_val = consis_loss.detach()
        contrast_loss_val = contrast_loss.detach()
        es_mean = entity_score.float().mean().item() if entity_score is not None else 0.0
        es_std = entity_score.float().std().item() if entity_score is not None else 0.0
        entity_push_val = entity_push.detach()
        return loss + consis_loss, nll_loss, consis_loss_val, es_mean, es_std, contrast_loss_val, entity_push_val

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        consis_loss_sum = sum(log.get('consis_loss', 0) for log in logging_outputs)
        es_mean_sum = sum(log.get('entity_score_mean', 0) for log in logging_outputs)
        es_std_sum = sum(log.get('entity_score_std', 0) for log in logging_outputs)
        contrast_loss_sum = sum(log.get('contrast_loss', 0) for log in logging_outputs)
        nlogs = len(logging_outputs) if logging_outputs else 1
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('consis_loss', consis_loss_sum / nlogs, 1, round=4)
        metrics.log_scalar('contrast_loss', contrast_loss_sum / nlogs, 1, round=4)
        metrics.log_scalar('entity_score_mean', es_mean_sum / nlogs, 1, round=4)
        metrics.log_scalar('entity_score_std', es_std_sum / nlogs, 1, round=4)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
