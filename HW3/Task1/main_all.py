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

from model import Encoder, DecoderAll, LuongAttnDecoderLength


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

train_data = pd.read_csv("/home/victai/SDML/HW3/Task1/data/hw3_1/all/train.csv", names=['input', 'target'])
train_data['pos'] = train_data['input'].apply(lambda x: x.split('EOS')[1].split('NOP')[0].strip().split(' '))
train_data['rhyme'] = train_data['input'].apply(lambda x: x.split('NOP')[1].split('NOE')[0].strip())
train_data['tgt_length'] = train_data['input'].apply(lambda x: int(x.split('NOE')[1].split('NOR')[0].strip()))
train_data['input'] = train_data['input'].apply(lambda x: x.split('EOS')[0].strip().split(' ')[1:])
train_data['target'] = train_data['target'].apply(lambda x: x.split(' ')[1:-1])
train_data['input_length'] = train_data['input'].apply(lambda x: len(x))
train_data = train_data[train_data.tgt_length < MAX_TGT_LEN]
train_data = train_data[train_data.input_length < MAX_INPUT_LEN]

# take 1000 samples for each length
reserved_idx = []
for i in range(1, MAX_TGT_LEN):
    reserved_idx.extend(train_data[train_data.tgt_length == i].index.values[:1000].tolist())
reserved_idx.sort()
train_data = train_data.iloc[reserved_idx]

input_words = np.array([b for a in train_data['input'].values for b in a])
target_words = np.array([b for a in train_data['target'].values for b in a])
all_words = np.unique(np.hstack((input_words, target_words)))

all_rhymes = train_data['rhyme'].drop_duplicates().values.tolist()
all_rhymes.sort(key=len, reverse=True)
all_rhymes = np.array(all_rhymes) # sort to prevent substring
all_rhymes = np.hstack((["Rhyme_PAD"], all_rhymes))

all_pos = np.unique(np.array([j for i in train_data['pos'].values.tolist() for j in i]))
all_pos = np.hstack((["POS_PAD"], all_pos))

rhyme_corpus = {}
for w in all_words:
    p_y = lazy_pinyin(w)[-1]
    for r in all_rhymes:
        if p_y.endswith(r):
            rhyme_corpus[w] = r
            break

all_words = np.hstack(([PAD_TOKEN], [SOS_TOKEN], [EOS_TOKEN], all_words))
num_words = len(all_words)

word2idx_mapping = {k: v for k, v in zip(all_words, np.arange(num_words))}
idx2word_mapping = {v: k for k, v in word2idx_mapping.items()}
rhyme2idx_mapping = {k: v for k, v in zip(all_rhymes, np.arange(len(all_rhymes)))}
idx2rhyme_mapping = {v: k for k, v in rhyme2idx_mapping.items()}
pos2idx_mapping = {k: v for k, v in zip(all_pos, np.arange(len(all_pos)))}
idx2pos_mapping = {v: k for k, v in pos2idx_mapping.items()}

train_data['input'] = train_data['input'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
train_data['target'] = train_data['target'].apply(lambda x: x + [EOS_TOKEN])  # add EOS at the end
train_data['input'] = train_data['input'].apply(lambda x: [word2idx_mapping[i] for i in x])
train_data['target'] = train_data['target'].apply(lambda x: [word2idx_mapping[i] for i in x])
train_data['input_length'] = train_data['input'].apply(lambda x: len(x))
train_data['target_length'] = train_data['target'].apply(lambda x: len(x))
train_data['target_pos'] = train_data['pos'].apply(lambda x: [pos2idx_mapping[i] for i in x])
train_data['target_rhyme'] = train_data['rhyme'].apply(lambda x: rhyme2idx_mapping[x])
input_lengths = train_data.input_length.values
target_lengths = train_data.tgt_length.values
target_pos = train_data.target_pos.values
target_rhyme = train_data.target_rhyme.values


train_data.reset_index(inplace=True)

for i, line in enumerate(target_pos):
    line = np.hstack((line, [pos2idx_mapping["POS_PAD"] for k in range(MAX_TGT_LEN-len(line))]))
    target_pos[i] = line
target_pos = np.array(target_pos.tolist())

train_data = list(zip(train_data.input, train_data.target))
for i, (a, b) in enumerate(train_data):
    a = np.hstack((a, [word2idx_mapping[PAD_TOKEN] for k in range(MAX_INPUT_LEN-len(a))]))
    b = np.hstack((b, [word2idx_mapping[PAD_TOKEN] for k in range(MAX_TGT_LEN-len(b))]))
    train_data[i] = (a, b)
train_data = np.array(train_data)

print("=== Process Data Done ==== Time cost {:.3f} secs".format(time.time() - t1))

## Process Data End

def predict(encoder, decoder):
    test_df = pd.read_csv(args.test_data, names=['input']).iloc[:3000]
    test_df['tgt_pos'] = test_df['input'].apply(lambda x: x.split('EOS')[1].split('NOP')[0].strip().split(' ')[:MAX_TGT_LEN])
    test_df['tgt_rhyme'] = test_df['input'].apply(lambda x: x.split('NOP')[1].split('NOE')[0].strip())
    test_df['tgt_length'] = test_df['input'].apply(lambda x: min(int(x.split('NOE')[1].split('NOR')[0].strip()), MAX_TGT_LEN-1))
    test_df['input'] = test_df['input'].apply(lambda x: x.split('EOS')[0].strip().split(' ')[1:MAX_INPUT_LEN+1] + [EOS_TOKEN])
    test_df['input'] = test_df['input'].apply(lambda x: [word2idx_mapping.get(i, 0) for i in x])
    test_df['tgt_pos'] = test_df['tgt_pos'].apply(lambda x: [pos2idx_mapping.get(i, 0) for i in x])
    test_df['mapped_tgt_rhyme'] = test_df['tgt_rhyme'].apply(lambda x: rhyme2idx_mapping.get(x, 0))
    tgt_pos = test_df.tgt_pos.values
    padded_tgt_pos = np.copy(tgt_pos)
    tgt_rhyme = test_df.tgt_rhyme.values
    mapped_tgt_rhyme = test_df.mapped_tgt_rhyme.values
    tgt_lengths = test_df.tgt_length.values
    max_length = max(tgt_lengths)
    test_df = test_df['input'].values

    for i in range(len(test_df)):
        test_df[i] = np.hstack((test_df[i], [word2idx_mapping[PAD_TOKEN] for k in range(MAX_INPUT_LEN - len(test_df[i]))]))
        padded_tgt_pos[i] = np.hstack((padded_tgt_pos[i], [pos2idx_mapping["POS_PAD"] for k in range(MAX_TGT_LEN - len(tgt_pos[i]))]))
    test_df = np.array(test_df)
    test_df = torch.LongTensor(test_df.tolist()).t().to(device)
    padded_tgt_pos = np.array(padded_tgt_pos.tolist())

    encoder = encoder.eval().to(device)
    decoder = decoder.eval().to(device)

    # predict
    batch_size = 1000
    result = [[] for i in range(max_length)]
    for b in tqdm(range(test_df.shape[1] // batch_size)):
        start = batch_size * b
        end = None if (b == (test_df.shape[1] // batch_size) - 1) else batch_size * (b+1)

        encoder_output, encoder_hidden = encoder(test_df[:, start: end])
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, encoder_output.shape[1]).to(device)
        batch_tgt_lengths = torch.from_numpy(tgt_lengths[start: end]).view(1, encoder_output.shape[1]).to(device)
        batch_tgt_pos = torch.from_numpy(padded_tgt_pos.T[:, start:end]).long().to(device)
        batch_tgt_rhyme = torch.from_numpy(mapped_tgt_rhyme[start: end]).view(1, encoder_output.shape[1]).to(device)

        for i in range(max_length):
            tmp_tgt_lengths = torch.clamp(batch_tgt_lengths - i, min=0)
            tmp_tgt_pos = batch_tgt_pos[i].unsqueeze(0)
            tmp_tgt_rhyme = batch_tgt_rhyme * (tmp_tgt_lengths <= 0).long()
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths, tmp_tgt_pos, tmp_tgt_rhyme)
                _, top1 = decoder_output.topk(70, dim=2)  # output top 70 for rhyme
                decoder_input = top1[:,:,0].squeeze(-1)
                result[i].extend(top1.cpu().numpy()[0].tolist())

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
                    for k, candidate in enumerate(last):
                        if j > 0 and rhyme_corpus.get(idx2word_mapping.get(candidate, 0), 0) == tgt_rhyme[i]:
                            new_result[i][j-1] = idx2word_mapping.get(candidate, 0)
                            cnt += 1
                            break

                    # calculate POS accuracy
                    all_pos_cnt += len(tgt_pos[i])    
                    if len(new_result[i]) > 0:
                        _, pos = zip(*jieba.posseg.cut(''.join(new_result[i])))
                        pos_len = min(len(pos), len(tgt_pos[i]))
                        pos = np.array([pos2idx_mapping.get(i, 0) for i in pos])
                        correct_pos_cnt += sum(np.array(pos[:pos_len]) == np.array(tgt_pos[i][:pos_len]))
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
    print("length: ", sum(res_len == tgt_lengths) / res_len.shape[0])

    output = subprocess.run(['python3', '/home/victai/SDML/HW3/trigram_model/trigram_evaluate.py', '--testing_data', args.output_file], stderr=subprocess.STDOUT)


def train(input_var, tgt_var, tgt_lengths, max_tgt_length, tgt_pos, tgt_rhyme, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, validate=False):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx_mapping[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    tgt_lengths = torch.from_numpy(tgt_lengths).view(1, input_var.shape[1])
    tgt_rhyme = torch.from_numpy(tgt_rhyme).view(1, input_var.shape[1])

    decoder_input = decoder_input.to(device)
    tgt_lengths = tgt_lengths.to(device)
    tgt_pos = tgt_pos.to(device)
    tgt_rhyme = tgt_rhyme.to(device)
    
    teacher_forcing_ratio = 0.5

    teacher_forcing = True if np.random.random() > teacher_forcing_ratio else False
    
    if teacher_forcing:
        for i in range(max_tgt_length):
            tmp_tgt_lengths = torch.clamp(tgt_lengths - i, min=0)
            input_pos = tgt_pos[i].unsqueeze(0)
            tgt_rhyme = tgt_rhyme * (tmp_tgt_lengths <= 0).long()
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths, input_pos, tgt_rhyme)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)

    else:
        for i in range(max_tgt_length):
            tmp_tgt_lengths = torch.clamp(tgt_lengths - i, min=0)
            input_pos = tgt_pos[i].unsqueeze(0)
            tgt_rhyme = tgt_rhyme * (tmp_tgt_lengths <= 0).long()
            if args.attn:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, tmp_tgt_lengths)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, tmp_tgt_lengths, input_pos, tgt_rhyme)
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

    return loss.item() / max_tgt_length / input_var.shape[0]

def main():
    epoch = 100
    batch_size = 64
    hidden_dim = 300

    encoder = Encoder(num_words, hidden_dim)
    if args.attn:
        attn_model = 'dot'
        decoder = LuongAttnDecoderLength(attn_model, hidden_dim, num_words, MAX_TGT_LEN)
    else:
        decoder = DecoderAll(hidden_dim, num_words, MAX_TGT_LEN, len(all_pos), len(all_rhymes))

    if args.train:
        weight = torch.ones(num_words)
        weight[word2idx_mapping[PAD_TOKEN]] = 0

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        weight = weight.to(device)

        encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
        decoder_optimizer = Adam(decoder.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=weight)


        np.random.seed(1124)
        order = np.arange(len(train_data))

        best_loss = 1e5
        best_epoch = 0
        
        for e in range(epoch):
            #if e - best_epoch > 20: break

            np.random.shuffle(order)
            shuffled_train_data = train_data[order]
            shuffled_y_lengths = target_lengths[order]
            shuffled_y_pos = target_pos[order]
            shuffled_y_rhyme = target_rhyme[order]
            train_loss = 0
            valid_loss = 0
            for b in tqdm(range(int(len(order) // batch_size))):
                batch_x = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 0].tolist()).t()
                batch_y = torch.LongTensor(shuffled_train_data[b*batch_size: (b+1)*batch_size][:, 1].tolist()).t()
                batch_y_lengths = shuffled_y_lengths[b*batch_size: (b+1)*batch_size]
                batch_y_pos = torch.LongTensor(shuffled_y_pos[b*batch_size: (b+1)*batch_size]).t()
                batch_y_rhyme = shuffled_y_rhyme[b*batch_size: (b+1)*batch_size]

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                train_loss += train(batch_x, batch_y, batch_y_lengths, max(batch_y_lengths), batch_y_pos, batch_y_rhyme, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, False)

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
