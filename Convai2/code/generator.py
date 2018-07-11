import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from parse_data_ver4 import *


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, pretrained_word_embed, gpu=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        #print("self.gpu =", self.gpu)
        #self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings = nn.Embedding.from_pretrained(pretrained_word_embed)
        #self.gru_encoder = nn.GRU(embedding_dim, hidden_dim)
        #self.gru_decoder = nn.GRU(embedding_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru_personas = nn.GRU(embedding_dim, hidden_dim)
        #self.gru2out_encoder = nn.Linear(hidden_dim, vocab_size)
        #self.gru2out_decoder = nn.Linear(hidden_dim, vocab_size)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)
        self.gru2out_personas = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        #if oracle_init:
        #    for p in self.parameters():
        #        init.normal(p, 0, 1)

    def init_hidden(self, personas):  #[batch_size, #sen, sen_len] -> [1, batch_size, hidden_dim]
        #h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        hidden = autograd.Variable(torch.zeros(personas.size(0), 1, personas.size(1), self.hidden_dim).cuda())
        personas = personas.permute(0, 2, 1)
        for i in range(personas.size(0)):                         #dialog
            for j in range(personas.size(1)):                     #sen_len
                out, hidden[i] = self.hidden_forward(personas[i][j], hidden[i].clone())   

        #hidden = [personas.size(0), 1, personas.size(1), self.hidden_dim]
        hidden = torch.squeeze(hidden, 1)  #[personas.size(0), personas.size(1), self.hidden_dim]
        hidden = torch.sum(hidden, 1, keepdim=True)      #[personas.size(0), 1, self.hidden_dim]
        hidden = hidden.permute(1, 0, 2)            #[1, personas.size(0), self.hidden_dim]
        #print("self.gpu in init_hidden =", self.gpu)
        #if self.gpu:
        #    print("hidden.cuda() =",hidden.cuda())
        #    return hidden.cuda()
        #else:
        return hidden

    def hidden_forward(self, inp, hidden):    #[#sen, sen_len](seq_len, batch, input_size) -> [#sen, hidden_dim]
        #print(inp.type())
        #print(hidden.type())
        #input dim                                                        # [#sent](batch_size)
        emb = self.embeddings(inp)                                        # [#sent, embed_dim](batch_size x embedding_dim)                
        emb = emb.view(1, -1, self.embedding_dim)                         # [1, #sent, embed_dim](1 x batch_size x embedding_dim)
        out, hidden = self.gru_personas(emb, hidden)                      # out:[1, #sent, hidden_dim](1 x batch_size x hidden_dim), hidden:[1, #sent, hidden_dim](num_layers * num_directions, batch, hidden_size)
        out = self.gru2out_personas(out.view(-1, self.hidden_dim))        # out:[#sent, hidden_dim](batch_size x vocab_size)
        out = F.log_softmax(out, 1)
        return out, hidden               # out:[#sent, hidden_dim](batch_size x vocab_size), hidden:[1, #sent, hidden_dim](num_layers * num_directions, batch, hidden_size)

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, 1)
        return out, hidden
    """
    def encoder_forward(self, inp, hidden):
        emb = self.embeddings(inp)
        emb = emb.view(1, -1, self.embedding_dim)
        out, hidden = self.gru_encoder(emb, hidden)
        out = self.gru2out_encoder(out.view(-1, self.hidden_dim))
        out = F.log_softmax(out)
        return out, hidden
    """
    """
    def decoder_forward(self, inp, hidden):
        '''
        Embeds input and applies GRU one token at a time (seq_len = 1)
        '''
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru_decoder(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out_decoder(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out)
        return out, hidden
    """
    def sample(self, episodes, start_letter=5):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(len(episodes), self.max_seq_len).type(torch.LongTensor)
        personas_your = get_persona_batch(episodes, 1)
        personas_partner = get_persona_batch(episodes, 0)
        #if self.gpu:
        #    personas_your = personas_your.cuda()
        #    personas_partner = personas_partner.cuda()
        turn_batch_list = get_dialog_batches(episodes)

        #batch_size, seq_len = inp.size()
        #inp = inp.permute(1, 0)           # seq_len x batch_size
        #target = target.permute(1, 0)     # seq_len x batch_size

        h_your = self.init_hidden(personas_your)
        h_partner = self.init_hidden(personas_partner)
        #print("h_your.type() =", h_your.type())
        #print("h_partner.type() =", h_partner.type())
        h = h_your+h_partner
        condition = h
        inp = autograd.Variable(torch.LongTensor([start_letter]*len(episodes)))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()
        
        loss = 0
        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            #print(out.data.size())
            samples[:, i] = out.data.view(-1)
            #h = self.init_hidden(num_samples)
            inp = out.view(-1)

        return samples, condition

    def batchNLLLoss(self, inp, target, personas_your, personas_partner):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        #if self.gpu:
        #    personas_your = personas_your.cuda()
        #    personas_partner = personas_partner.cuda()

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        #print("8")
        h_your = self.init_hidden(personas_your)
        #print("9")
        h_partner = self.init_hidden(personas_partner)
        #print("10")
        #print(h_your.type())
        #print(h_partner.type())
        h = h_your+h_partner
        #print(h.type())
        #print("11")
        
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            #print(out.type())
            #print(h.type())
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward, condition):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        #h = self.init_hidden(batch_size)
        h = condition

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

