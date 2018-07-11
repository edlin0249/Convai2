import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
from random import uniform
from torch.nn.parameter import Parameter

class Seq2Seq(nn.Module):
    def __init__(self, opt):
        super(Seq2Seq, self).__init__()
        self.symbal = {'PAD': 0, 'BOS': 1, 'EOS': 2, 'UNK': 3}
        self.opt = opt
        self.histories = None
        self.personas = None #Variable(torch.Tensor([]))
        self.lt = nn.Embedding(self.opt['num_embeddings'], self.opt['embedding_dim'])
        if type(self.opt['embdding_weight']) != type(None):
            self.lt.weight = Parameter(self.opt['embdding_weight'])
        #self.lt.requires_grad = False
        # encoder
        self.encoder = self.opt['rnn_class'](self.opt['embedding_dim'], self.opt['hidden_size'], self.opt['num_layers'], 
                                             dropout=self.opt['dropout'], bidirectional=self.opt['bidirectional'], batch_first=True)
        # decoder
        self.decoder_hidden_size = self.opt['hidden_size'] * (2 if self.opt['bidirectional'] else 1)
        self.decoder_input_size = self.opt['embedding_dim'] + self.opt['history_size'] + self.opt['persona_size']
        self.decoder = self.opt['rnn_class'](self.decoder_input_size, self.decoder_hidden_size, self.opt['num_layers'],
                                             dropout=self.opt['dropout'], batch_first=True)
        # decoder_output to word_embedding
        self.o2e = nn.Linear(self.decoder_hidden_size, self.opt['embedding_dim'], bias=False)
        # word_embedding to vocab
        self.e2v = nn.Linear(self.opt['embedding_dim'], self.opt['num_embeddings'], bias=False)
        # attention
        self.history_attention = Attention(self.opt['history_size'] + self.decoder_hidden_size, 1).cuda()
        self.words_attention   = Attention(self.opt['embedding_dim'] + self.decoder_hidden_size, 1).cuda()
        self.persona_attention = Attention(self.opt['persona_size'] + self.decoder_hidden_size, 1).cuda()

    def select_history(self, select):
        assert(type(select) == int)
        if type(self.histories) != type(None):
            batch_size = self.histories.size(0)
            h = self.histories[select]
            self.histories = h.expand(batch_size, *tuple(h.size()))

    def init_persona(self, personas):
        # persona: batch_size, num_sentence, num_words
        if type(self.personas) != type(None):
            del self.personas
        if type(self.histories) != type(None):
            del self.histories
        batch_size = personas.size(0)
        num_sentence = personas.size(1)
        num_words = personas.size(2)

        personas = personas.view(-1, num_words)
        # need to embedding again every step
        personas = self.lt(personas)
        
        self.personas = personas.view(batch_size, num_sentence, num_words, self.opt['embedding_dim'])
        self.histories = None

    def forward(self, xs, ys):
        # xs: batch_size, num_words | type=int 
        # ys: batch_size, num_words | type=int
        #print ('Encoder begin!')
        batch_size = xs.size(0)
        h0 = Variable(torch.zeros(self.opt['num_layers'] * self.opt['bidirectional'], batch_size, self.opt['hidden_size']), requires_grad=False)
        x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
        xs = self.lt(xs)
        try:
            x_lens, indices = torch.sort(Variable(torch.LongTensor(x_lens).cuda()))
            _, recover_indices = torch.sort(indices)
            xs = xs[indices]
            xs = pack_padded_sequence(xs, x_lens.data.int().tolist(), batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack the
            pass
        # lt(xs): batch_size, num_words, embedding_dim
        _ , hidden_status = self.encoder(xs)
        # hidden_status: num_layers * bidirectional, batch_size, hidden_size
        #print (type(hidden_status[0]), type(recover_indices))
        hidden_status = hidden_status[:, recover_indices]
        hidden_status = hidden_status.transpose(0, 1).contiguous().view(self.opt['num_layers'], batch_size, -1)
        history = hidden_status
        BOS = torch.cat([Variable(torch.LongTensor([self.symbal['BOS']]).cuda().unsqueeze(0))] * batch_size, 0)
        preds = []
        outputs = []
        ##########
        #  Test  #
        ##########
        if type(ys) == type(None):
            #y = BOS
            #for _ in range(self.opt['longest_label']):
            # y = self.lt()
                
            pass
                
        ##########
        # Train  #
        ##########
        else:
            ys = torch.cat((BOS, ys), 1)
            for i in range(ys.size(1) - 1):
                # y: 1 dimension
                y = ys[:, i].unsqueeze(1)
                y = self.lt(y)
                #print ('Attention begin!')
                sentence_vectors = self.words_attention(self.personas, hidden_status[-1])
                persona_vectors  = self.persona_attention(sentence_vectors, hidden_status[-1])
                if type(self.histories) == type(None):
                    history_vectors = Variable(torch.zeros(batch_size, self.opt['history_size'])).cuda()
                else:
                    #print ('history_attention')
                    #print ('histories size = ', torch.cat(self.histories, 1).size())
                    history_vectors = self.history_attention(self.histories, hidden_status[-1])
                #print (type(y), type(history_vectors), type(persona_vectors))
                input = torch.cat((y, history_vectors.unsqueeze(1), persona_vectors.unsqueeze(1)), 2)
                
                #print ('Decoder begin!')
                decoder_output, hidden_status = self.decoder(input, hidden_status)
                # decoder_output: batch_size, 1, history_size
                embedding = self.o2e(decoder_output)
                # embedding: batch_size, 1, embedding_dim
                vocab = self.e2v(embedding)
                # vocab: batch_size, 1, num_embeddings
                #print ('vocab size = {}'.format(vocab.size()))
                pred = vocab.data.max(2, keepdim=True)[1]
                preds.append(pred)
                #print ('pred type = {}, pred size = {}'.format(type(pred), pred.size()))
                #print ('vocab type = {}, vocab size = {}'.format(type(vocab), vocab.size()))
                outputs.append(vocab)

        history = torch.cat((history, hidden_status), 2).transpose(0, 1).contiguous()
        #print ('history_size = ', history.size())
        if type(self.histories) == type(None):
            self.histories = history
        else:
            self.histories = torch.cat((self.histories, history), 1)
        preds = torch.cat(preds, 1)
        outputs = torch.cat(outputs, 1)
        #print (outputs)
        #print ('preds type = {}, outputs type = {}'.format(type(preds), type(outputs)))
        #print ('preds size = {}, outputs size = {}'.format(preds.size(), outputs.size()))

        return preds, outputs


class Attention(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention = nn.Linear(self.input_size, self.output_size)

    def forward(self, input, hidden_status):
        # input: batch_size, number of history, history_size or 
        #          batch_size, number of persona words, embedding_dim or 
        #         batch_size, number of persona, persona_size

        # hidden_status: batch_size, hidden_size
        
        d = len(input.size())
        Index = 1
        while d > len(hidden_status.size()):
            hidden_status = torch.cat([hidden_status.unsqueeze(Index)] * input.size(Index), Index)
            Index += 1
        #print ('input size = ', input.size())
        #print ('hidden_status size = ', hidden_status.size())
        #print (self.attention)
        alpha = self.attention(torch.cat((input, hidden_status), d - 1))
        alpha = F.softmax(alpha, dim=d-2)
        output = input * alpha
        output = torch.sum(output, d - 2)

        return output
