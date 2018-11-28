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

from model import Encoder, DecoderLength, LuongAttnDecoderLength

parser = argparse.ArgumentParser()
parser.add_argument("--create_data", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--attn", action="store_true", default=False)
parser.add_argument("--test_data", type=str, default="../data/ta_input.txt")
parser.add_argument("--output_file", type=str, default="result.txt")
parser.add_argument("--encoder_path", type=str, default="encoder.pt")
parser.add_argument("--decoder_path", type=str, default="decoder.pt")
parser.add_argument("--use_cuda", action="store_true", default=True)
args = parser.parse_args()

PAD_TOKEN = 'PAD' #0
SOS_TOKEN = 'SOS' #1
EOS_TOKEN = 'EOS' #2

MAX_LEN = 31

## Process Data Section

print("Processing Data")
t1 = time.time()

train_data = pd.read_csv("/home/victai/SDML/HW3/Task1/data/hw3_1/length/train.csv", names=['input', 'target'])
train_data['input'] = train_data['input'].apply(lambda x: x.split(' '))
train_data['tgt_length'] = train_data['input'].apply(lambda x: int(x[-2]))
train_data['input'] = train_data['input'].apply(lambda x: x[1:-3])
train_data['target'] = train_data['target'].apply(lambda x: x.split(' ')[1:-1])
train_data['input_length'] = train_data['input'].apply(lambda x: len(x))
train_data = train_data[train_data.tgt_length < MAX_LEN]
train_data = train_data[train_data.input_length < 50]
reserved_idx = []
for i in range(1, MAX_LEN):
    reserved_idx.extend(train_data[train_data.tgt_length == i].index.values[:100].tolist())
reserved_idx.sort()
train_data = train_data.iloc[reserved_idx]


input_words = np.array([b for a in train_data['input'].values for b in a])
target_words = np.array([b for a in train_data['target'].values for b in a])
all_words = np.unique(np.hstack((input_words, target_words)))
all_words = np.hstack(([PAD_TOKEN], [SOS_TOKEN], [EOS_TOKEN], all_words))
num_words = len(all_words)

word2idx_mapping = {k: v for k, v in zip(all_words, np.arange(num_words))}
idx2word_mapping = {v: k for k, v in word2idx_mapping.items()}

train_data['input'] = train_data['input'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
train_data['target'] = train_data['target'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
train_data['input'] = train_data['input'].apply(lambda x: [word2idx_mapping[i] for i in x])
train_data['target'] = train_data['target'].apply(lambda x: [word2idx_mapping[i] for i in x])
train_data['input_length'] = train_data['input'].apply(lambda x: len(x))
train_data['target_length'] = train_data['target'].apply(lambda x: len(x))
input_lengths = train_data.input_length.values
target_lengths = train_data.tgt_length.values

max_input_length, max_target_length = MAX_LEN, MAX_LEN
train_data.reset_index(inplace=True)

train_data = list(zip(train_data.input, train_data.target))
for i, (a, b) in enumerate(train_data):
    a = np.hstack((a, [word2idx_mapping[PAD_TOKEN] for k in range(max_input_length-len(a))]))
    b = np.hstack((b, [word2idx_mapping[PAD_TOKEN] for k in range(max_target_length-len(b))]))
    train_data[i] = (a, b)
train_data = np.array(train_data)

print("=== Process Data Done ==== Time cost {:.3f} secs".format(time.time() - t1))

## Process Data End

def predict(encoder, decoder):
    test_df = pd.read_csv("/home/victai/SDML/HW3/Task1/data/hw3_1/length/test.csv", names=['input'])
    test_df['input'] = test_df['input'].apply(lambda x: x.split(' '))
    test_df['tgt_length'] = test_df['input'].apply(lambda x: min(int(x[-2]), MAX_LEN-1))
    test_df['input'] = test_df['input'].apply(lambda x: x[1:-3])
    test_df['input'] = test_df['input'].apply(lambda x: x[:50] + [EOS_TOKEN])
    test_df['input'] = test_df['input'].apply(lambda x: [word2idx_mapping.get(i, 0) for i in x])
    tgt_lengths = test_df.tgt_length.values
    max_length = max(tgt_lengths)
    test_df = test_df['input'].values

    for i in range(len(test_df)):
        test_df[i] = np.hstack((test_df[i], [word2idx_mapping[PAD_TOKEN] for k in range(50 - len(test_df[i]))]))
    test_df = np.array(test_df)
    test_df = torch.LongTensor(test_df.tolist()).t().cuda()

    encoder = encoder.eval().cuda()
    decoder = decoder.eval().cuda()

    batch_size = 1000
    result = [[] for i in range(max_length)]
    for b in range(test_df.shape[1] // batch_size):
        print(b, '\r', end='')
        start = batch_size * b
        end = None if (b == (test_df.shape[1] // batch_size) - 1) else batch_size * (b+1)
        encoder_output, encoder_hidden = encoder(test_df[:, start: end])
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, encoder_output.shape[1]).cuda()
        batch_tgt_lengths = torch.from_numpy(tgt_lengths[start: end]).view(1, encoder_output.shape[1]).cuda()
        for i in range(max_length):
            tmp_tgt_lengths = torch.clamp(batch_tgt_lengths - i, min=0)
            if args.attn:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths)
                _, top1 = decoder_output.topk(1, dim=2)
                decoder_input = top1.squeeze(-1)
            result[i].extend(decoder_input.cpu().numpy()[0].tolist())
    #result = np.transpose(np.array(result), (1,0,2))
    result = np.array(result).T
    res_len = np.zeros(result.shape[0])
    with open(args.output_file, "w") as f:
        for i, res in enumerate(result):
            for j, w in enumerate(res):
                if (idx2word_mapping[w] == EOS_TOKEN) or (j == len(res)-1):
                    res_len[i] = j
                    print("", file=f)
                    break
                print(idx2word_mapping[w], end=' ', file=f)
    ipdb.set_trace()
    print("accuracy: ", sum(res_len == tgt_lengths) / res_len.shape[0])

def train(input_var, tgt_var, tgt_lengths, max_tgt_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda=False, validate=False):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    tgt_lengths = torch.from_numpy(tgt_lengths).view(1, input_var.shape[1])
    if use_cuda:
        decoder_input = decoder_input.cuda()
        tgt_lengths = tgt_lengths.cuda()
    
    teacher_forcing_ratio = 0.5

    teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False
    
    if teacher_forcing:
        for i in range(max_tgt_length):
            tmp_tgt_lengths = torch.clamp(tgt_lengths - i, min=0)
            if args.attn:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)
        #print("encode time:[{:.3f}], en_de time:[{:.3f}], teacher decode time[{:.3f}]".format(b-a, c-b, d_t-c))

    else:
        for i in range(max_tgt_length):
            tmp_tgt_lengths = torch.clamp(tgt_lengths - i, min=0)
            if args.attn:
                decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths)
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
        decoder = LuongAttnDecoderLength(attn_model, hidden_dim, num_words, MAX_LEN)
    else:
        decoder = DecoderLength(hidden_dim, num_words, MAX_LEN)

    if args.train:
        weight = torch.ones(num_words)
        weight[word2idx_mapping[PAD_TOKEN]] = 0
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            weight = weight.cuda()
        encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
        decoder_optimizer = Adam(decoder.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=weight)


        np.random.seed(1124)
        order = np.arange(len(train_data))

        best_loss = 1e10
        best_epoch = 0
        
        for e in range(epoch):
            #if e - best_epoch > 20: break

            np.random.shuffle(order)
            shuffled_train_data = train_data[order]
            shuffled_x_lengths = input_lengths[order]
            shuffled_y_lengths = target_lengths[order]
            train_loss = 0
            valid_loss = 0
            for b in tqdm(range(int(len(order) // batch_size))):
                #print(b, '\r', end='')
                batch_x = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()
                batch_x_lengths = shuffled_x_lengths[b*batch_size: (b+1)*batch_size]
                batch_y_lengths = shuffled_y_lengths[b*batch_size: (b+1)*batch_size]

                if use_cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                train_loss += train(batch_x, batch_y, batch_y_lengths, max(batch_y_lengths), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_cuda, False)

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
        if use_cuda:
            encoder.load_state_dict(torch.load(args.encoder_path))
            decoder.load_state_dict(torch.load(args.decoder_path))
        else:
            encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device('cpu')))
        print(encoder)
        print(decoder)


    predict(encoder, decoder)
    #submit()

if __name__ == '__main__':
    main()
    


# Reference Paper: https://aclweb.org/anthology/D16-1140
