import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class LSTM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, dropout=0.2, tie_weights=False):
        """
        @params
            ntoken: int, dictionary size
            ninp: int, word embedding size
            nhid: list/array, each entry indicates the hidden size
        """
        super(LSTM, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid[-1], ntoken)
        self.nhid = nhid
        self.nlayers = len(nhid)

        self.lstm = []
        size = [ninp] + nhid
        for i in range(self.nlayers):
            self.lstm.append(nn.LSTMCell(size[i], size[i+1]))

        self.init_weights()
        # self.bn1 = nn.BatchNorm1d(nhid[0])

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):
        """
        @params
            input: intput word vector, one-hot coding
            hidden: tuple, (hx, cx), both have the size of nlayers * batch_size * nhidden

        @return
            output
            updated hidden
        """
        input_ = self.encoder(input)
        input_ = self.drop(input_)
        hid = []
        output = []
        for t in range(input_.size(0)):
            hx = input_[t]
            for i in range(self.nlayers):
                hx, cx = self.lstm[i](hx, hidden[i])
                # hx, cx = self.lstm[i](self.drop(hx), hidden[i])
                hid.append((hx, cx))
            hx = self.drop(hx)
            y = F.log_softmax(self.decoder(hx))
            output.append(y)
        output = T.stack(output, 1)
        return output, hid

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hid = []
        for i in range(self.nlayers):
            hx = Variable(weight.new(bsz, self.nhid[i]).zero_())
            cx = Variable(weight.new(bsz, self.nhid[i]).zero_())
            hid.append((hx, cx))
        return hid
