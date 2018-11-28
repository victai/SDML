import numpy as np
import pandas as pd
import ipdb
import pickle
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Encoder, Decoder, LuongAttnDecoderRNN

parser = argparse.ArgumentParser()
parser.add_argument("--create_data", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--attn", action="store_true", default=False)
parser.add_argument("--test_data", type=str, default="../data/ta_input.txt")
parser.add_argument("--output_file", type=str, default="result.txt")
parser.add_argument("--encoder_path", type=str, default="encoder.pt")
parser.add_argument("--decoder_path", type=str, default="decoder.pt")
args = parser.parse_args()

PAD_TOKEN = '<PAD>' #0
SOS_TOKEN = '<SOS>' #1
EOS_TOKEN = '<EOS>' #2

cut_length = 20 # cut sentences longer than 20 segments

## Process Data Section

print("Processing Data")
t1 = time.time()

data = pd.read_csv("/home/victai/SDML/HW3/data/hw_train.csv")
data['lyric_segments'] = data['lyric_segments'].apply(lambda x: x.split(' ')[:cut_length])
song_ids = data['lyric no'].drop_duplicates().values


all_words = np.unique(np.array([b for a in data['lyric_segments'].values for b in a]))
all_words = np.hstack(([PAD_TOKEN], [SOS_TOKEN], [EOS_TOKEN], all_words))
num_words = len(all_words)

word2idx_mapping = {k: v for k, v in zip(all_words, np.arange(num_words))}
idx2word_mapping = {v: k for k, v in word2idx_mapping.items()}

data['lyric_segments'] = data['lyric_segments'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
data['mapped_segments'] = data['lyric_segments'].apply(lambda x: [word2idx_mapping[i] for i in x])

max_seqlen = 0
for i in data['lyric_segments']:
    max_seqlen = max(max_seqlen, len(i))


def gen_data(df):
    df.reset_index(inplace=True)
    last_lyrics = sorted(df['lyric no'].drop_duplicates(keep='last').index, reverse=True)
    ret_data = list(zip(df['mapped_segments'][:-1], df['mapped_segments'][1:]))

    for i in last_lyrics[1:]:   
        del ret_data[i]
    ret_data = np.array(ret_data)

    x_lengths = np.array([len(i) for i, j in ret_data])
    y_lengths = np.array([len(j) for i, j in ret_data])

    for i, (a, b) in enumerate(ret_data):
        a = np.hstack((a, [word2idx_mapping[PAD_TOKEN] for k in range(max_seqlen-len(a))]))
        b = np.hstack((b, [word2idx_mapping[PAD_TOKEN] for k in range(max_seqlen-len(b))]))
        ret_data[i] = (a, b)
    return ret_data, x_lengths, y_lengths

train_song_ids = song_ids[:2000]
valid_song_ids = song_ids[2000: 2500]
#data = data[data['lyric no'].isin(song_ids)]

train_data, train_x_lengths, train_y_lengths = gen_data(data[data['lyric no'].isin(train_song_ids)])
valid_data, valid_x_lengths, valid_y_lengths = gen_data(data[data['lyric no'].isin(valid_song_ids)])


print("=== Process Data Done ==== Time cost {:.3f} secs".format(time.time() - t1))

## Process Data End


def predict(test_path, encoder, decoder):
    with open(test_path, "r") as f:
        test_data = f.read().split('\n')[:-1]
    for i in range(len(test_data)):
        test_data[i] = test_data[i].split(" ")[:cut_length] + [EOS_TOKEN]
        for j, w in enumerate(test_data[i]):
            try:
                test_data[i][j] = word2idx_mapping[w]
            except:
                test_data[i][j] = word2idx_mapping[PAD_TOKEN]
    lengths = np.array([len(i) for i in test_data])
    max_length = max(lengths)
    length_order = np.argsort(-lengths)

    for i in range(len(test_data)):
        test_data[i] = np.hstack((test_data[i], [word2idx_mapping[PAD_TOKEN] for k in range(max_length - len(test_data[i]))]))
    test_data = np.array(test_data)
    test_data = torch.LongTensor(test_data.tolist())[length_order].t()

    encoder.eval().cpu()
    decoder.eval().cpu()

    encoder_output, encoder_hidden = encoder(test_data)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, test_data.shape[1])
    result = []
    for i in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        _, top1 = decoder_output.topk(1, dim=2)
        decoder_input = top1.squeeze(-1)
        result.append(decoder_input.numpy()[0])
    #result = np.transpose(np.array(result), (1,0,2))
    result = np.array(result).T
    with open(args.output_file, "w") as f:
        for i in result:
            for j, w in enumerate(i):
                if (idx2word_mapping[w] == EOS_TOKEN) or (j == len(i)-1):
                    print("", file=f)
                    break
                print(idx2word_mapping[w], end=' ', file=f)

def train(input_var, tgt_var, max_tgt_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda=False, validate=False):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    if use_cuda:
        decoder_input = decoder_input.cuda()
    
    teacher_forcing_ratio = 0.5

    teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False
    
    if teacher_forcing:
        for i in range(max_tgt_length):
            if args.attn:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)
        #print("encode time:[{:.3f}], en_de time:[{:.3f}], teacher decode time[{:.3f}]".format(b-a, c-b, d_t-c))

    else:
        for i in range(max_tgt_length):
            if args.attn:
                decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            if args.attn:
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                _, top1 = decoder_output.topk(1, dim=2)
                decoder_input = top1.squeeze(-1)
        #print("encode time:[{:.3f}], en_de time:[{:.3f}], normal decode time[{:.3f}]".format(b-a, c-b, d-c))

    if not validate:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / max_tgt_length / input_var.shape[0]

def main():
    epoch = 1000
    batch_size = 64
    hidden_dim = 300
    use_cuda = True

    encoder = Encoder(num_words, hidden_dim)
    if args.attn:
        attn_model = 'dot'
        decoder = LuongAttnDecoderRNN(attn_model, hidden_dim, num_words)
    else:
        decoder = Decoder(hidden_dim, num_words)

    if args.train:
        weight = torch.ones(num_words)
        weight[word2idx_mapping[PAD_TOKEN]] = 0
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            weight = weight.cuda()
        encoder_optimizer = Adam(encoder.parameters(), lr=0.003)
        decoder_optimizer = Adam(decoder.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss(weight=weight)


        np.random.seed(1124)
        order = np.arange(len(train_data))

        best_loss = 1e10
        best_epoch = 0
        
        for e in range(epoch):
            #if e - best_epoch > 20: break

            np.random.shuffle(order)
            shuffled_train_data = train_data[order]
            shuffled_y_lengths = train_y_lengths[order]
            train_loss = 0
            valid_loss = 0
            for b in tqdm(range(int(len(order) // batch_size))):
                #print(b, '\r', end='')
                batch_x = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()
                batch_y_lengths = shuffled_y_lengths[b*batch_size: (b+1)*batch_size]

                if use_cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                train_loss += train(batch_x, batch_y, max_seqlen, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda, False)

            train_loss /= b

            '''
            for b in range(len(valid_data) // batch_size):
                batch_x = torch.LongTensor(valid_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(valid_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()
                if use_cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                valid_loss += train(batch_x, batch_y, max_seqlen, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda, True)
            valid_loss /= b
            '''
            print("epoch {}, train_loss {:.4f}, valid_loss {:.4f}, best_epoch {}, best_loss {:.4f}".format(e, train_loss, valid_loss, best_epoch, best_loss))

            '''
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = e
                torch.save(encoder.state_dict(), args.encoder_path + '.best')
                torch.save(decoder.state_dict(), args.decoder_path + '.best')
            '''
            torch.save(encoder.state_dict(), args.encoder_path)
            torch.save(decoder.state_dict(), args.decoder_path)
        print(encoder)
        print(decoder)
        print("==============")

    else:
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))
        print(encoder)
        print(decoder)


    predict(args.test_data, encoder, decoder)
    #submit()

if __name__ == '__main__':
    main()
    
