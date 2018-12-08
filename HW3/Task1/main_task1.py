import numpy as np
import pandas as pd
import ipdb
import pickle
import argparse
import time
from tqdm import tqdm
from pypinyin import lazy_pinyin
import jieba.posseg
import subprocess

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Encoder, DecoderTask1, LuongAttnDecoderLength


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--attn", action="store_true", default=False)
parser.add_argument("--test_data", type=str, default="../data/ta_input.txt")
parser.add_argument("--output_file", type=str, default="result.txt")
parser.add_argument("--encoder_path", type=str, default="encoder.pt")
parser.add_argument("--decoder_path", type=str, default="decoder.pt")
parser.add_argument("--use_cuda", action="store_true", default=False)
args = parser.parse_args()

device = 'cpu'
if args.use_cuda:
    device = 'cuda'

PAD_TOKEN = 'PAD' #0
SOS_TOKEN = 'SOS' #1
EOS_TOKEN = 'EOS' #2

MAX_TGT_LEN = 31
MAX_INPUT_LEN = 50

## Process Data Section

print("Processing Data")
t1 = time.time()

train_df = pd.read_csv("/home/victai/SDML/HW3/Task1/data/hw3_1/all/train.csv", names=['input', 'target'])
train_df['target_pos'] = train_df['input'].apply(lambda x: x.split('EOS')[1].split('NOP')[0].strip().split(' '))
train_df['target_rhyme'] = train_df['input'].apply(lambda x: x.split('NOP')[1].split('NOE')[0].strip())
train_df['target_length'] = train_df['input'].apply(lambda x: int(x.split('NOE')[1].split('NOR')[0].strip()))
train_df['input'] = train_df['input'].apply(lambda x: x.split('EOS')[0].strip().split(' ')[1:])
train_df['target'] = train_df['target'].apply(lambda x: x.split(' ')[1:-1])
train_df['input_length'] = train_df['input'].apply(lambda x: len(x))
train_df = train_df[train_df['target_length'] < MAX_TGT_LEN]
train_df = train_df[train_df['input_length'] < MAX_INPUT_LEN]

# take 1000 samples for each length
reserved_idx = []
for i in range(1, MAX_TGT_LEN):
    reserved_idx.extend(train_df[train_df['target_length'] == i].index.values[:5000].tolist())
reserved_idx.sort()
train_df = train_df.iloc[reserved_idx]

input_words = np.array([b for a in train_df['input'].values for b in a])
target_words = np.array([b for a in train_df['target'].values for b in a])
all_words = np.unique(np.hstack((input_words, target_words)))

all_rhymes = train_df['target_rhyme'].drop_duplicates().values.tolist()
all_rhymes.sort(key=len, reverse=True)
all_rhymes = np.array(all_rhymes) # sort to prevent substring
all_rhymes = np.hstack((["RHYME_PAD"], all_rhymes))

all_pos = np.unique(np.array([j for i in train_df['target_pos'].values.tolist() for j in i]))
all_pos = np.hstack((["POS_PAD"], all_pos))

rhyme_corpus = {}
for w in all_words:
    p_y = lazy_pinyin(w)[-1]
    for r in all_rhymes:
        if p_y.endswith(r):
            rhyme_corpus[w] = r
            break

all_words = pd.unique(np.hstack(([PAD_TOKEN], [SOS_TOKEN], [EOS_TOKEN], all_words, ['NOP', 'NOE', 'NOR'], all_rhymes, all_pos, np.arange(1, MAX_TGT_LEN+1))))
num_words = len(all_words)

word2idx_mapping = {k: v for k, v in zip(all_words, np.arange(num_words))}
idx2word_mapping = {v: k for k, v in word2idx_mapping.items()}
rhyme2idx_mapping = {k: v for k, v in zip(all_rhymes, np.arange(len(all_rhymes)))}
idx2rhyme_mapping = {v: k for k, v in rhyme2idx_mapping.items()}
pos2idx_mapping = {k: v for k, v in zip(all_pos, np.arange(len(all_pos)))}
idx2pos_mapping = {v: k for k, v in pos2idx_mapping.items()}

train_df['input'] = train_df['input'].apply(lambda x: x + [EOS_TOKEN] + [PAD_TOKEN for i in range(MAX_INPUT_LEN-len(x)-1)])
train_df['target'] = train_df['target'].apply(lambda x: x + [EOS_TOKEN] + [PAD_TOKEN for i in range(MAX_TGT_LEN-len(x)-1)])
train_df['target_length'] = train_df['target_length'].apply(lambda x: [str(x), "NOR"])
train_df['target_pos'] = train_df['target_pos'].apply(lambda x: x + ["POS_PAD" for i in range(MAX_TGT_LEN-len(x))] + ["NOP"])
train_df['target_rhyme'] = train_df['target_rhyme'].apply(lambda x: [x, "NOE"])
train_df['input'] = train_df['input'] + train_df["target_pos"] + train_df["target_rhyme"] + train_df["target_length"]
train_df['input'] = train_df['input'].apply(lambda x: [word2idx_mapping[i] for i in x])
train_df['target'] = train_df['target'].apply(lambda x: [word2idx_mapping[i] for i in x])

train_data = np.array(list(zip(train_df.input, train_df.target)))
for i, (a, b) in enumerate(train_data):
    train_data[i] = (np.array(a), np.array(b))

print("=== Process Data Done ==== Time cost {:.3f} secs".format(time.time() - t1))

## Process Data End

def predict(encoder, decoder):
    test_df = pd.read_csv(args.test_data, names=['input'])

    test_df['target_pos'] = test_df['input'].apply(lambda x: x.split('EOS')[1].split('NOP')[0].strip().split(' ')[:MAX_TGT_LEN])
    test_df['target_rhyme'] = test_df['input'].apply(lambda x: x.split('NOP')[1].split('NOE')[0].strip())
    test_df['target_length'] = test_df['input'].apply(lambda x: x.split('NOE')[1].split('NOR')[0].strip())
    test_df['input'] = test_df['input'].apply(lambda x: x.split('EOS')[0].strip().split(' ')[1:MAX_INPUT_LEN+1])
    test_df['input'] = test_df['input'].apply(lambda x: x + [EOS_TOKEN] + [PAD_TOKEN for i in range(MAX_INPUT_LEN-len(x)-1)])  # add EOS at the end
    target_rhyme = np.copy(test_df.target_rhyme.values)
    target_length = np.copy(test_df.target_length.values.astype(int))
    target_pos = np.copy(test_df.target_pos.values)

    test_df['target_rhyme'] = test_df['target_rhyme'].apply(lambda x: [x, "NOE"])
    test_df['target_length'] = test_df['target_length'].apply(lambda x: [x, "NOR"])
    test_df['target_pos'] = test_df['target_pos'].apply(lambda x: x + ["POS_PAD" for i in range(MAX_TGT_LEN-len(x))] + ["NOP"])
    test_df['input'] = test_df['input'] + test_df["target_pos"] + test_df["target_rhyme"] + test_df["target_length"]
    test_df['input'] = test_df['input'].apply(lambda x: [word2idx_mapping.get(i, 0) for i in x])

    test_data = test_df['input'].values

    test_data = torch.LongTensor(test_data.tolist()).t().to(device)

    encoder = encoder.eval().to(device)
    decoder = decoder.eval().to(device)

    # predict
    batch_size = 1000
    result = [[] for i in range(MAX_TGT_LEN)]
    for b in tqdm(range(test_data.shape[1] // batch_size)):
        start = batch_size * b
        end = None if (b == (test_data.shape[1] // batch_size) - 1) else batch_size * (b+1)

        encoder_output, encoder_hidden = encoder(test_data[:, start: end])
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, encoder_output.shape[1]).to(device)

        for i in range(MAX_TGT_LEN):
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                _, top1 = decoder_output.topk(1, dim=2)
                decoder_input = top1.squeeze(-1)
                result[i].extend(top1.cpu().numpy()[0].tolist())
                #_, top1 = decoder_output.topk(70, dim=2)  # output top 70 for rhyme
                #decoder_input = top1[:,:,0].squeeze(-1)
                #result[i].extend(top1.cpu().numpy()[0].tolist())

    ipdb.set_trace()
    # output
    result = np.transpose(np.array(result), (1,0,2))
    res_len = np.zeros(result.shape[0])
    new_result = [[] for i in range(result.shape[0])]
    cnt = 0
    all_pos_cnt = 0
    correct_pos_cnt = 0
    with open(args.output_file, "w") as f:
        for i, res in enumerate(result):
            for j, w in enumerate(res):
                if (idx2word_mapping[w[0]] == EOS_TOKEN) or (j == len(res)-1):
                    res_len[i] = j

                    # pick correct rhyme
                    #for k, candidate in enumerate(last):
                    #    if j > 0 and rhyme_corpus.get(idx2word_mapping.get(candidate, 0), 0) == tgt_rhyme[i]:
                    #        new_result[i][j-1] = idx2word_mapping.get(candidate, 0)
                    #        cnt += 1
                    #        break
                    if j > 0 and rhyme_corpus.get(idx2word_mapping.get(last[0], 0), 0) == target_rhyme[i]:
                        cnt += 1

                    # calculate POS accuracy
                    all_pos_cnt += len(target_pos[i])    
                    if len(new_result[i]) > 0:
                        _, pos = zip(*jieba.posseg.cut(''.join(new_result[i])))
                        pos_len = min(len(pos), len(target_pos[i]))
                        #pos = np.array([pos2idx_mapping.get(i, 0) for i in pos])
                        correct_pos_cnt += sum(np.array(pos[:pos_len]) == np.array(target_pos[i][:pos_len]))
                    else:
                        print("", file=f)
                        break

                    for k, c in enumerate(new_result[i]):
                        if k == len(new_result[i]) - 1:
                            print(c, file=f)
                        else:
                            print(c, end=' ', file=f)
                    break
                new_result[i].append(idx2word_mapping[w[0]])
                last = w  # save last timestep's output
    print('pos:', correct_pos_cnt / all_pos_cnt)
    print('rhyme:', cnt/result.shape[0])
    print("length: ", sum(res_len == target_length) / res_len.shape[0])

    output = subprocess.run(['python3', '/home/victai/SDML/HW3/trigram_model/trigram_evaluate.py', '--testing_data', args.output_file], stderr=subprocess.STDOUT)


def train(input_var, tgt_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, validate=False):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, input_var.shape[1])

    decoder_input = decoder_input.to(device)
    
    teacher_forcing_ratio = 0.5

    teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False
    
    if teacher_forcing:
        for i in range(MAX_TGT_LEN):
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)

    else:
        for i in range(MAX_TGT_LEN):
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            if args.attn:
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                _, top1 = decoder_output.topk(1, dim=2)
                decoder_input = top1.squeeze(-1)

    if not validate:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / MAX_TGT_LEN / input_var.shape[0]

def main():
    epoch = 1000
    batch_size = 64
    hidden_dim = 256

    encoder = Encoder(num_words, hidden_dim)
    if args.attn:
        attn_model = 'dot'
        decoder = LuongAttnDecoderLength(attn_model, hidden_dim, num_words, MAX_TGT_LEN)
    else:
        decoder = DecoderTask1(hidden_dim, num_words)

    if args.train:
        weight = torch.ones(num_words)
        weight[word2idx_mapping[PAD_TOKEN]] = 0

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        weight = weight.to(device)

        encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
        decoder_optimizer = Adam(decoder.parameters(), lr=0.001)
        criterion = nn.NLLLoss(ignore_index=int(word2idx_mapping[PAD_TOKEN]), size_average=True)
        #criterion = nn.CrossEntropyLoss(weight=weight)


        np.random.seed(1124)
        order = np.arange(len(train_data))

        best_loss = 1e5
        best_epoch = 0
        
        for e in range(epoch):
            #if e - best_epoch > 20: break

            #np.random.shuffle(order)
            choice = np.random.choice(order, 10000, replace=False)
            shuffled_train_data = train_data[choice]
            train_loss = 0
            valid_loss = 0
            for b in tqdm(range(int(len(choice) // batch_size))):
                batch_x = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                train_loss += train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, False)

            train_loss /= b

            '''
            for b in range(len(valid_data) // batch_size):
                batch_x = torch.LongTensor(valid_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(valid_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                valid_loss += train(batch_x, batch_y, max_seqlen, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, True)
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

    else:
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device(device)))
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device(device)))
        print(encoder)
        print(decoder)
    print("==========================================================")

    predict(encoder, decoder)

if __name__ == '__main__':
    main()
    

# Reference Paper: https://aclweb.org/anthology/D16-1140
