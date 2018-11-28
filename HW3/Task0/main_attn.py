import numpy as np
import pandas as pd
import ipdb
import pickle
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Encoder, LuongAttnDecoderRNN

parser = argparse.ArgumentParser()
parser.add_argument("--create_data", action="store_true", default=False)
args = parser.parse_args()

## Process Data Section
PAD_TOKEN = '<PAD>' #0
SOS_TOKEN = '<SOS>' #1
EOS_TOKEN = '<EOS>' #2

data = pd.read_csv("/home/victai/SDML/HW3/data/hw_train.csv")
data['lyric_segments'] = data['lyric_segments'].apply(lambda x: x.split(' ')[:50])

M = 0
for i in data['lyric_segments']:
    M = max(M, len(i))
max_seqlen = M+1

all_words = np.unique(np.array([b for a in data['lyric_segments'].values for b in a]))
all_words = np.hstack(([PAD_TOKEN], [SOS_TOKEN], [EOS_TOKEN], all_words))
num_words = len(all_words)

word2idx_mapping = {k: v for k, v in zip(all_words, np.arange(num_words))}
idx2word_mapping = {v: k for k, v in word2idx_mapping.items()}

data['lyric_segments'] = data['lyric_segments'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
data['mapped_segments'] = data['lyric_segments'].apply(lambda x: [word2idx_mapping[i] for i in x])

last_lyrics = sorted(data['lyric no'].drop_duplicates(keep='last').index, reverse=True)

#if args.create_data:
train_data = list(zip(data['mapped_segments'][:-1], data['mapped_segments'][1:]))

for i in last_lyrics[1:]:
    del train_data[i]
train_data = np.array(train_data)

#for i, (a, b) in enumerate(train_data):
#    train_data[i] = (np.hstack(([word2idx_mapping[PAD_TOKEN] for k in range(max_seqlen-1-len(a))], a, word2idx_mapping[EOS_TOKEN])), b)

#    with open("train_data.pkl", "wb") as f:
#        pickle.dump(train_data, f)
#else:
#    with open("train_data.pkl", "rb") as f:
#        train_data = pickle.load(f)



## Process Data End

use_cuda = True


def predict(test_path, encoder, decoder):
    with open(test_path, "r") as f:
        test_data = f.read().split('\n')[:-1]
    for i in range(len(test_data)):
        test_data[i] = test_data[i].split(" ")[:50] + [EOS_TOKEN]
        for j, w in enumerate(test_data[i]):
            test_data[i][j] = word2idx_mapping[w]
    lengths = np.array([len(i) for i in test_data])
    max_length = max(lengths)
    length_order = np.argsort(-lengths)

    for i in range(len(test_data)):
        test_data[i] = np.hstack((test_data[i], [word2idx_mapping[PAD_TOKEN] for k in range(max_length - len(test_data[i]))]))
    test_data = np.array(test_data)
    test_data = torch.LongTensor(test_data.tolist())[length_order].t()

    encoder.eval().cpu()
    decoder.eval().cpu()

    encoder_output, encoder_hidden = encoder(test_data, lengths[length_order])
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, test_data.shape[1])
    result = []
    for i in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        try:
            _, top1 = decoder_output.topk(1, dim=2)
        except:
            _, top1 = decoder_output.topk(1, dim=1)
            top1 = top1.unsqueeze(0)
        decoder_input = top1.squeeze(-1)
        result.append(decoder_input.numpy())
    result = np.array(result).T.reshape(len(test_data), max_length)
    with open("result.txt", "w") as f:
        for i in result:
            for j in i:
                if idx2word_mapping[j] == EOS_TOKEN:
                    print("\n", file=f)
                    break
                print(idx2word_mapping[j], end=' ', file=f)




def train(input_var, tgt_var, input_lengths, max_tgt_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda=False):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var, input_lengths)

    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    if use_cuda:
        decoder_input = decoder_input.cuda()

    teacher_forcing_ratio = 0.5

    teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False
    

    #ipdb.set_trace()
    if teacher_forcing:
        for i in range(max_tgt_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)

    else:
        for i in range(max_tgt_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            try:
                _, top1 = decoder_output.topk(1, dim=2)
            except:
                _, top1 = decoder_output.topk(1, dim=1)
                top1 = top1.unsqueeze(0)
            decoder_input = top1.squeeze(-1)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_seqlen / input_var.shape[0]

def main():
    global train_data
    epoch = 2
    batch_size = 60
    hidden_dim = 10
    use_cuda = True

    encoder = Encoder(num_words, hidden_dim)
    decoder = LuongAttnDecoderRNN('dot', hidden_dim, num_words, use_cuda=use_cuda)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    np.random.seed(1124)
    order = np.arange(len(train_data))
   
    for e in range(epoch):
        np.random.shuffle(order)
        train_data = train_data[order]
        loss = 0
        for b in range(len(order) // batch_size):
            print(b, '\r', end='')
            batch_x = train_data[b*batch_size: (b+1)*batch_size][:, 0]
            batch_y = train_data[b*batch_size: (b+1)*batch_size][:, 1]
            x_length = np.array([len(x) for x in batch_x])
            max_x_length = max(x_length)
            length_order = np.argsort(-x_length)
            for i in range(batch_size):
                batch_x[i] = np.hstack((batch_x[i], [word2idx_mapping[PAD_TOKEN] for k in range(max_x_length - len(batch_x[i]))]))
            batch_x = torch.LongTensor(batch_x.tolist())[length_order].t()
            
            y_length = [len(y) for y in batch_y]
            #max_y_length = max(y_length)
            max_y_length = min(max(y_length), 51)
            for i in range(batch_size):
                batch_y[i] = np.hstack((batch_y[i], [word2idx_mapping[PAD_TOKEN] for k in range(max_y_length - len(batch_y[i]))]))
            try:
                batch_y = torch.LongTensor(batch_y.tolist())[length_order].t()
            except:
                ipdb.set_trace()

            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            loss += train(batch_x, batch_y, x_length[length_order], max_y_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda)

        loss /= b
        print("epoch {}, loss {:.4f}".format(e, loss))
    
    torch.save(encoder.state_dict(), "encoder_attn.pt")
    torch.save(decoder.state_dict(), "decoder_attn.pt")

    #encoder.load_state_dict(torch.load("encoder.pt"))
    #decoder.load_state_dict(torch.load("decoder.pt"))


    predict("../data/ta_input.txt", encoder, decoder)
    #submit()

if __name__ == '__main__':
    main()
    
