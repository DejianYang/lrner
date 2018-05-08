# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from charbilstm import CharBiLSTM
from charbigru import CharBiGRU
from charcnn import CharCNN
from model.TransBiLSTM import TransBiLSTM


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print "build word representation..."
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.use_trans = data.use_trans
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        self.w = nn.Linear(data.word_emb_dim, data.HP_trans_hidden_dim)

        if self.use_trans:
            self.trans_hidden_dim = data.HP_trans_hidden_dim
            self.trans_embedding_dim = data.trans_emb_dim
            self.trans_feature = TransBiLSTM(data.translation_alphabet.size(), self.trans_embedding_dim,
                                             self.trans_hidden_dim,
                                             data.HP_dropout, data.pretrain_trans_embedding, self.gpu)

        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_seq_feature == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim,
                                            data.HP_dropout, self.gpu)
            elif data.char_seq_feature == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim,
                                               data.HP_dropout, data.pretrain_char_embedding, self.gpu)
            elif data.char_seq_feature == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim,
                                              data.HP_dropout, self.gpu)
            elif data.char_seq_feature == "ALL":
                self.char_all_feature = True
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim,
                                            data.HP_dropout, self.gpu)
                self.char_feature_extra = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim,
                                                     self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print "Error char feature selection, please check parameter data.char_seq_feature (CNN/LSTM/GRU/ALL)."
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        self.feature_num = data.feature_num
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(
                nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]))
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
            else:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(
                    self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                trans_inputs, trans_seq_length, trans_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]

        for idx in range(self.feature_num):
            word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))

        if self.use_char:
            # calculate char lstm last hidden
            char_features, _ = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            # concat word and char together
            word_list.append(char_features)
            # word_embs = torch.cat([word_embs, char_features], 2)
            if self.char_all_feature:
                char_features_extra, _ = self.char_feature_extra.get_last_hiddens(char_inputs,
                                                                                  char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                # concat word and char together
                word_list.append(char_features_extra)

        if self.use_trans:
            trans_features, trans_rnn_length = self.trans_feature.get_last_hiddens(trans_inputs,
                                                                                   trans_seq_length.cpu().numpy())

            trans_features_wc = trans_features
            if self.gpu:
                trans_features_wc.cuda()
            trans_features_wc = trans_features_wc[trans_seq_recover]
            trans_inputs = trans_inputs[trans_seq_recover]
            word_embs_temp = word_embs.view(batch_size * sent_len, -1)
            for index, line in enumerate(trans_inputs):
                if line[0].data.cpu().numpy()[0] == 0:
                    trans_features_wc[index] = self.w(word_embs_temp[index])

            trans_features_wc_temp = trans_features_wc
            trans_features_wc = trans_features_wc.view(batch_size, sent_len, -1)

            word_list.append(trans_features_wc)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent, self.w(word_embs_temp), trans_features_wc_temp
