# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --label-smoothing=<float>               use label smoothing [default: 0.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm

from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0.):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = input_feed

        # initialize neural network layers...

        """ 
            2(a)
                Initialize source and target embeddings.
                    self.src_embed: Embedding layer for source language
                    self.tgt_embed: Embedding layer for target language

            Hints: 
                1. Using torch.nn.Embedding function to initialize embeddings.
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                2. `vocab` object contains two vocabularies:
                        `vocab.src` for source
                        `vocab.tgt` for target
                3. You can get the length of a specific vocabulary by running:
                        `len(vocab.<specific_vocabulary>)`
                4. When creating the Embedding layer, specify the padding token idx in the argument.
        """

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### WRITE YOUR CODE HERE (~2 Lines)

        self.src_embed = 
        self.tgt_embed = 

        ### END OF YOUR CODE HERE

        """ 
        2(b)
            Initialize the following variables:
                self.encoder_lstm: Bidirectional LSTM with bias. Note that the hidden state of biLSTM is 2*hidden_size, which is the concatenation of forward and backward hidden states.
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                self.decoder_lstm: LSTM Cell with bias. Note that when input_feed is True, the decoder uses the input-feeding approach in Sec.3.3 of Luong et al.
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
                self.decoder_cell_init: Linear. Initialize the decoder's cell with the encoder's last time step cell
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                self.decoder_state_init: Linear. Initialize the decoder's state with the encoder's last time step states
                self.att_src_linear: Linear with no bias. This transfers the hidden states from encoder bi-LSTM into the same dimension with the decoder state
                self.att_vec_linear: Linear with no bias. This is W_c in Luong et al.
                self.readout: Linear Layer with no bias. This is W_s in Luong et al.
                self.dropout: Dropout_layer
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        """

        ### WRITE YOUR CODE HERE (~8 Lines)

        self.encoder_lstm = 
        self.decoder_lstm = 
        self.att_src_linear = 
        self.att_vec_linear = 
        self.readout = 
        self.dropout = 
        self.decoder_cell_init = 
        self.decoder_state_init = 

        ### END OF YOUR CODE HERE

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt),
                                                           padding_idx=vocab.tgt['<pad>'])

    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (src_sent_len, batch_size)
        src_sents_var = self.vocab.src.to_input_tensor(src_sents, device=self.device)
        # (tgt_sent_len, batch_size)
        tgt_sents_var = self.vocab.tgt.to_input_tensor(tgt_sents, device=self.device)
        src_sents_len = [len(s) for s in src_sents]

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

        if self.label_smoothing:
            # (tgt_sent_len - 1, batch_size)
            tgt_gold_words_log_prob = self.label_smoothing_loss(
                tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
        else:
            # (tgt_sent_len, batch_size)
            tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

            # (tgt_sent_len - 1, batch_size)
            tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1),
                                                   dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # (batch_size)
        scores = tgt_gold_words_log_prob.sum(dim=0)

        return scores

    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(self.device)

    def encode(self, src_sents_var: torch.Tensor, src_sent_lens: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Use biLSTM to encode source sentences into hidden states.
        Besides, according to the final states of the encoder to obtain initial states for decoder.

        Args:
            src_sents_var: Tensor of padded source sentences with shape (src_len_max, batch_size),
                            where src_len_max is the maximum source sentence length.
            src_sent_lens: List of actual lengths for each of the source sentences in the batch.

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                            with shape (batch_size, source_sentence_length, hidden_size * 2)
            decoder_init_state: decoder LSTM's initial state, computed from source encodings.
                                It is the tuple of tensors, tuple(Tensor, Tensor), representing the decoder's initial
                                hidden state and cell.
        Steps:
            1. Construct Tensor `src_word_embeds` of source sentences with shape (src_sent_len, batch_size, embed_size)
                using the source embedding that you initialized in __init__.
                Note that there is no initial hidden state or cell for the encoder.
            2. Compute `src_encodings`, `last_state`, `last_cell` by applying the encoder to `src_word_embeds`.
                - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to `src_word_embeds`.
                        refer: The input section of https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
                        refer: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
                - Note that the shape of the tensor returned by the encoder is (src_sent_len, batch_size, hidden_size * 2)
                    and we want to return a tensor of shape (batch_size, src_sent_len, hidden_size * 2) as `src_encodings`.
            3. Compute `dec_init_state` and `dec_init_cell`, which are the initial state and cell of the decoder.
                - `dec_init_state`:
                    output `h_n` (See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) from encoder_lstm is a
                    tensor shaped (2, batch_size, hidden_size).  The first dimension corresponds to forward and backward.
                    Concatenate the forwards and backwards tensors to obtain a tensor shape (batch_size, 2*hidden_size).
                    Apply the decoder_state_init layer to this in order to compute dec_init_state.
                - `dec_init_cell`:
                    output `c_n` (See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) from encoder_lstm is a
                    tensor shaped (2, batch_size, hidden_size). The first dimension corresponds to forward and backward.
                    Concatenate the forwards and backwards tensors to obtain a tensor shape (batch_size, 2*hidden_size).
                    Apply the decoder_cell_init layer to this in order to compute dec_init_cell.
        """

        ### WRITE YOUR CODE HERE (~7 Lines)


        ### END OF YOUR CODE HERE

        return src_encodings, (dec_init_state, dec_init_cell)

    def decode(self, src_encodings: torch.Tensor, src_sent_masks: torch.Tensor,
               decoder_init_vec: Tuple[torch.Tensor, torch.Tensor], tgt_sents_var: torch.Tensor) -> torch.Tensor:
        """
        Given source encodings, compute output hidden representations to predict the target sequence

        Args:
            src_encodings: Hidden states of tokens in source sentences. The shape is (batch_size, src_sent_len, hidden_size*2)
            src_sent_masks: Tensor of sentence masks. The shape is (batch_size, src_sent_len)
            decoder_init_state: Decoder LSTM's initial state. (tuple(Tensor, Tensor))
            tgt_sents: List of gold-standard target sentences, wrapped by `<s>` and `</s>`. The shape is (tgt_sent_len-1, batch_size)

        Returns:
            att_ves: The attentional hidden state from the decoder, with shape (tgt_sent_len-1, batch_size, hidden_size). This is \tilde{h}_t in Luong et al.

        Steps:
            1. Apply the attention projection layer (att_src_linear) to `src_encodings` to obtain `src_encoding_att_linear`,
            which should be shape (batch_size, src_sent_len, hidden_size).
            2. Construct tensor `tgt_word_embeds` of target sentences with shape (tgt_sent_len-1, batch_size, embed_size)
                using the targe embeddings self.tgt_embed.
            3. Use the torch.split function to iterate over the time dimension of `tgt_word_embeds`.
                In each step of the loop, you would have a splitted `y_tm1_embed` from `tgt_word_embeds` of shape (1, batch size, embedding size).
                    - Squeeze `y_tm1_embed` into a tensor of dimension (batch size, embedding size).
                    - If input_feed: Construct `x` by concatenating `y_tm1_embed` with `att_tm1`.
                    - Use the self.step() function to compute the Decoder's next (cell, state) values as well as the new
                        attention vector att_t.
                    - Append att_t to att_ves
                    - Update att_tm1 to the new att_t.
            4. Use torch.stack to convert att_ves from a list length tgt_sent_len of tensors all with shape (batch_size, hidden_size),
                to a single tensor with shape (tgt_sent_len-1, batch_size, hidden_size).
        """

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        h_tm1 = decoder_init_vec

        # Initialize a list we will use to collect the combined output att_t on each step
        att_ves = []

        ### WRITE YOUR CODE HERE (~13 Lines)


        ### END OF YOUR CODE HERE

        return att_ves

    def step(self, x: torch.Tensor,
             h_tm1: Tuple[torch.Tensor, torch.Tensor],
             src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor, src_sent_masks: torch.Tensor) -> Tuple[
        Tuple, torch.Tensor, torch.Tensor]:
        """
        Perform a single time step decoding in the decoder.

        Args:
            x (Tensor): The input for decoder. Concatenated Tensor of [y_tm1_embed, att_tm1],
                with shape (batch_size, embedding_size + hidden_size)
            h_tm1 (tuple(Tensor, Tensor)): Decoder state (hidden and cell).Tuple od tensor both with shape (batch_size, hidden_size)
            src_encodings (Tensor): Encoder hidden states Tensor, with shape (batch_size, src_sent_len, hidden_size * 2)
            src_encoding_att_linear (Tensor): Projected from (hidden_size * 2) to hidden_size.
                                                The shape is (batch_size, src_sent_len, hidden_size)
            src_sent_masks (Tensor): Tensor of sentence masks shape (batch_size, src_sent_len)

        Returns:
            decoder_state (h_t, cell_t) (tuple (Tensor, Tensor)): New decoder's state. Tuple of tensors both shape (batch_size, hidden_size)
                        First tensor is decoder's new hidden state, second tensor is decoder's new cell.
            att_t (Tensor): Attention vector at timestep t. The shape is (batch_size, hidden_size)
            alpha_t (Tensor): Tensor of shape (batch_size, src_sent_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                        We are simply returning this value so that we can sanity check
                                        your implementation.

        Steps:
            1. Apply the decoder to `x` and `h_tm1` to obtain the new decoder state (h_t, cell_t) at timestep t.
            2. Call self.dot_prod_attention() to obtain context_vector at timestep t ctx_t
                and alpha_t.
            3. Compute attention weight at timestep t by Equation.5 in Luong et al.:
                att_t = tanh(W_c[h_t; ctx_t]), where W_c refers to att_vec_linear, [;] refers to concatenation.
            4. Apply dropout layer to att_t.
        """

        ### WRITE YOUR CODE HERE (~4 Lines)


        ### END OF YOUR CODE HERE

        return (h_t, cell_t), att_t, alpha_t

    def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform dot attention in Luong et al.

        Args:
            h_t (Tensor): (batch_size, hidden_size) New hidden state for decoder
            src_encoding (Tensor): Encoder hidden states Tensor, with shape (batch_size, src_sent_len, hidden_size * 2)
            src_encoding_att_linear (Tensor): Projected from (hidden_size * 2) to hidden_size.
                                                The shape is (batch_size, src_sent_len, hidden_size)
            mask (Tensor): Tensor of sentence masks shape (batch_size, src_sent_len)

        Returns:
            ctx_vec (Tensor): The source-side contex vector. The shape is (batch_size, hidden_size * 2)
            softmaxed_att_weight (Tensor): The shape is (batch_size, src_sent_len)

        Step 1. Compute attention scores att_weight, a Tensor shape (batch_size, src_sent_len), which is score(h_t, \bar{h}_s) in Luong et al.
        """

        ### WRITE YOUR CODE HERE (~1 line)


        ### END OF YOUR CODE HERE

        if mask is not None:
            att_weight.data.masked_fill_(mask.bool(), -float('inf'))

        """
        Step 2. Apply softmax to att_weight to yield softmaxed_att_weight
        Step 3. Use batched matrix multiplication between softmaxed_att_weight and src_encoding to obtain the
                context vector, ctx_vec, with size (batch_size, hidden_size * 2).
        """

        ### WRITE YOUR CODE HERE (~3 lines)


        ### END OF YOUR CODE HERE

        return ctx_vec, softmaxed_att_weight

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear,
                                                      src_sent_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[int(prev_hyp_id)] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def sample(self, src_sents: List[List[str]], sample_size=5, max_decoding_time_step=100) -> List[Hypothesis]:
        """
        Given a batched list of source sentences, randomly sample hypotheses from the model distribution p(y|x)

        Args:
            src_sents: a list of batched source sentences
            sample_size: sample size for each source sentence in the batch
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var = self.vocab.src.to_input_tensor(src_sents, self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(sent) for sent in src_sents])
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec

        batch_size = len(src_sents)
        total_sample_size = sample_size * len(src_sents)

        # (total_sample_size, max_src_len, src_encoding_size)
        src_encodings = src_encodings.repeat(sample_size, 1, 1)
        src_encodings_att_linear = src_encodings_att_linear.repeat(sample_size, 1, 1)

        src_sent_masks = self.get_attention_mask(src_encodings,
                                                 [len(sent) for _ in range(sample_size) for sent in src_sents])

        h_tm1 = (h_tm1[0].repeat(sample_size, 1), h_tm1[1].repeat(sample_size, 1))

        att_tm1 = torch.zeros(total_sample_size, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device=self.device)
        sample_scores = torch.zeros(total_sample_size, device=self.device)

        samples = [torch.tensor([self.vocab.tgt['<s>']] * total_sample_size, dtype=torch.long, device=self.device)]

        t = 0
        while t < max_decoding_time_step:
            t += 1

            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            # probabilities over target words
            p_t = F.softmax(self.readout(att_t), dim=-1)
            log_p_t = torch.log(p_t)

            # (total_sample_size)
            y_t = torch.multinomial(p_t, num_samples=1)
            log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
            y_t = y_t.squeeze(1)

            samples.append(y_t)

            sample_ends |= torch.eq(y_t, eos_id).byte()
            sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())

            if torch.all(sample_ends):
                break

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        for t, y_t in enumerate(samples):
            for i, sampled_word_id in enumerate(y_t):
                sampled_word_id = sampled_word_id.cpu().item()
                src_sent_id = i % batch_size
                sample_id = i // batch_size

                if t == 0 or _completed_samples[src_sent_id][sample_id][-1] != eos_id:
                    _completed_samples[src_sent_id][sample_id].append(sampled_word_id)

        completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
        for src_sent_id in range(batch_size):
            for sample_id in range(sample_size):
                offset = sample_id * batch_size + src_sent_id
                hyp = Hypothesis(value=self.vocab.tgt.indices2words(_completed_samples[src_sent_id][sample_id])[:-1],
                                 score=sample_scores[offset].item())
                completed_samples[src_sent_id][sample_id] = hyp

        return completed_samples

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


def evaluate_ppl(model, dev_data, batch_size=32):
    """
    Evaluate perplexity on dev sentences

    Args:
        dev_data: a list of dev sentences
        batch_size: batch size

    Returns:
        ppl: the perplexity on dev sentences
    """

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # you may want to wrap the following code using a context manager provided
    # by the NN library to signal the backend to not to keep gradient information
    # e.g., `torch.no_grad()`

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=args['--input-feed'],
                label_smoothing=float(args['--label-smoothing']),
                vocab=vocab)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (
                                                                                                 time.time() - train_time),
                                                                                         time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cum_tgt_words),
                                                                                             cum_examples),
                      file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[
    List[Hypothesis]]:
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    print(f"load test source sentences from [{args['TEST_SOURCE_FILE']}]", file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print(f"load test target sentences from [{args['TEST_TARGET_FILE']}]", file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()
