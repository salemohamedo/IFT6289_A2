import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def input_transpose(sents, pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch, and transpose the resulted sentences.

    Args:
        sents: (list[list[str]]): list of tokenized sentences, where each sentence
                                    is represented as a list of words
        pad_token: (str): padding token

    Returns:
        sents_padded: (list[list[str]]): list of padded and transposed sentences, where each element in this list
                                            should be a list of length len(sents), containing the ith token in each
                                            sentence. Sentences shorter than the max length sentence are padded out with
                                            the pad_token, such that each sentences in the batch now has equal length.
    """

    sents_padded = []

    ### WRITE YOUR CODE HERE (~5 lines)


    ### END OF YOUR CODE HERE

    return sents_padded


def read_corpus(file_path, source):
    data = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            sent = line.strip().split(' ')
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                sent = ['<s>'] + sent + ['</s>']
            data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """

    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss
