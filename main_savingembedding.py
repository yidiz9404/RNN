import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import data
import model
### to shuffle training data ###
from torch.utils.data import DataLoader
import pickle

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=str, default="50",
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--voc-size', type=int,  default=10000,
                    help='vocabulary size')
parser.add_argument('--L2', type=int,  default=0.001,
                    help='ridge regression parameter')
parser.add_argument('--cell', type=int,  default=0,
                    help='whether or not use rnn cell module')
parser.add_argument('--plot', type=str,  default='plot.png',
                    help='path for Tsne plot')
parser.add_argument('--saveembed', type=str,  default='embedding.pk',
                    help='path for embedding')
args = parser.parse_args()

###############################################################################
# Output the language model configuration
###############################################################################
nhidden = [int(i) for i in args.nhid.split(",")]
print "=" * 10 + "Language Model Configuration" + "=" * 10 + "\n"
if args.cell:
    print "use rnn cell"
    nlayers = len(nhidden)
else:
    print "use rnn only"
    nlayers = args.nlayers
print "--data\t\t{}\n--model\t\t{}".format(args.data, args.model)
print "--emsize\t{}\n--nhid\t\t{}".format(args.emsize, args.nhid)
print "--nlayers\t{}\n--lr\t\t{}".format(nlayers, args.lr)
print "--clip\t\t{}\n--epochs\t{}".format(args.clip, args.epochs)
print "--batch-size\t{}\n--bptt\t\t{}".format(args.batch_size, args.bptt)
print "--seed\t\t{}\n--log-interval\t{}".format(args.seed, args.log_interval)
print "--voc-size\t{}\n--L2\t\t{}".format(args.voc_size, args.L2)
print "--dropout\t{}\n--cell\t\t{}".format(args.dropout, args.cell)
print "\n" + "=" * 50 + "\n"

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.voc_size)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.cell:
    model = model.LSTM(ntokens, args.emsize, nhidden, args.dropout, args.tied)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, nhidden[0], args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2

    totalnorm = math.sqrt(totalnorm)
    #if totalnorm > clip:
    #    for p in model.parameters():
    #        p.grad.data = p.grad.data * clip / (totalnorm + 1e-6)
    return min(1, args.clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        clipped_lr = lr * clip_gradient(model, args.clip)

        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)

        #clip_gradient(model, args.clip)
        #opt.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
prev_val_loss = None
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    # To Shuffle the training data #
 #   train_loader = DataLoader(train_data,batch_size=len(train_data),shuffle=True)
 #   train_data = iter(train_loader).next()

    train()
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
    prev_val_loss = val_loss


# Run on test data and save the model.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
if args.save != '':
    with open(args.save, 'wb') as f:
        torch.save(model, f)

##Saving Embedding ###
embeddings = model.encoder
#Embeddings_to_plot = embeddings.weight.data.numpy()
Embeddings_to_plot = embeddings.weight.data.cpu().numpy()
if args.saveembed != '':
    with open(args.saveembed, 'wb') as f_2:
        pickle.dum(embeddings_to_plot, f_2)

