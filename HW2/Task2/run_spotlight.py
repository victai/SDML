import numpy as np
import pandas as pd
import time, pickle
import sys, time
import datetime
import torch
import argparse

sys.path.append("./spotlight")
from spotlight.cross_validation import user_based_train_test_split, timestamp_based_train_test_split
from spotlight.interactions import Interactions
from spotlight.evaluation import sequence_mrr_score, AveragePrecision, NewAveragePrecision
from spotlight.sequence.implicit import ImplicitSequenceModel
from torch.optim import SparseAdam, Adam

def get_data(args):
    if args.create_data:
        print("Creating Dataset")
        df_rate = pd.read_csv("data/rating_train.csv")
        df_rate.date = pd.to_datetime(df_rate.date)
        df_rate.foodid += 1
        user_ids = df_rate.userid.drop_duplicates().values
        min_date = {}
        for u in user_ids:
            min_date[u] = df_rate[df_rate.userid == u].date.min()
        def f(x):
            return (x['date'] - min_date[x['userid']]).days

        df_rate['timestamp'] = df_rate.apply(f, axis=1)

        df_rate['mapped_userid'] = df_rate.userid
        df_rate['mapped_userid'] = df_rate['mapped_userid'].apply(lambda x: np.where(user_ids == x)[0][0])

        cnt = np.zeros((2608, 5533))
        norm_cnt = np.zeros((2608, 5533))
        neg_prob = np.zeros((2608, 5533))
        for i, u in enumerate(user_ids):
            for f in df_rate[df_rate.userid == u].foodid:
                cnt[i][f] += 1
            norm_cnt[i] = cnt[i] / cnt[i].max()
            neg_prob[i] = cnt[i] / cnt[i].sum()   # higher cnt -> lower probability

        aaa = map(lambda x, y: cnt[np.where(user_ids==x)[0][0]][y], df_rate.userid, df_rate.foodid)
        bbb = map(lambda x, y: norm_cnt[np.where(user_ids==x)[0][0]][y], df_rate.userid, df_rate.foodid)
        df_rate['cnt'] = list(aaa)
        df_rate['norm_cnt'] = list(bbb)

        df_rate.to_csv("data/new_rating_train.csv", index=False)
        np.save("neg_prob.npy", neg_prob)
    else:
        print("Loadaing Dataset")
        df_rate = pd.read_csv("data/new_rating_train.csv")
        user_ids = df_rate.userid.drop_duplicates().values
        neg_prob = np.load("neg_prob.npy")

    return df_rate, user_ids, neg_prob


def predict(args, model, df_rate, neg_prob):
    user_ids = df_rate.userid.drop_duplicates().values
    answer = []
    for j, u in enumerate(user_ids):
        cur_data = df_rate[df_rate.userid == u]
        pred = -model.predict(np.expand_dims(cur_data.iloc[-args.seq_len:].foodid.values, axis=0))[0]
        pred[np.where(neg_prob[j] == 0)[0]] = 1e10
        answer.append(np.argsort(pred[1:])[:20])

    with open("submission.csv", "w") as f:
        f.write("userid,foodid\n")
        for i, u in enumerate(df_rate.userid.drop_duplicates()):
            f.write("{},".format(u))
            for j in answer[i]:
                f.write("{} ".format(j))
            f.write('\n')

    #!kaggle competitions submit -c ntucsie-sdml2018-2-2 -f submission.csv -m "spotlight"
    #time.sleep(5)
    #!kaggle competitions submissions ntucsie-sdml2018-2-2 | more"

def main(args):
    df_rate, user_ids, neg_prob = get_data(args)

    load_model = False

    print("seq_len {}, epoch {}".format(args.seq_len, args.epoch))

    dataset = Interactions(df_rate.mapped_userid.values.astype("int32"), \
                           df_rate.foodid.values.astype("int32"), \
                           timestamps=df_rate.timestamp.values.astype("int32"), \
                           weights=df_rate.norm_cnt.values.astype("float32"))

    train, test = timestamp_based_train_test_split(dataset, test_percentage=0.2)
    train = train.to_sequence(max_sequence_length=args.seq_len, min_sequence_length=args.min_seq_len, step_size=args.step_size, mode=args.mode)
    test = test.to_sequence(max_sequence_length=args.seq_len, min_sequence_length=args.min_seq_len, step_size=args.step_size, mode=args.mode)
    dataset = dataset.to_sequence(max_sequence_length=args.seq_len, min_sequence_length=args.min_seq_len, step_size=args.step_size, mode=args.mode)

    model = ImplicitSequenceModel(loss=args.loss,
                                  representation=args.representation, 
                                  embedding_dim=args.embedding_dim, 
                                  n_iter=args.epoch, 
                                  batch_size=256, 
                                  l2=0.0, 
                                  learning_rate=0.001, 
                                  optimizer_func=None, 
                                  use_cuda=True, 
                                  sparse=False, 
                                  random_state=None, 
                                  num_negative_samples=args.num_negative_samples,
                                  test_data=test,
                                  neg_prob=neg_prob)

    print("train.shape", train.sequences.shape)
    print("test.shape", test.sequences.shape)
    print("Fitting model")
    for i in range(5):
        model.fit(dataset, verbose=True, calc_map=args.calc_map, neg_mode=args.neg_mode)
    model.fit(test, verbose=True, calc_map=args.calc_map, neg_mode=args.neg_mode)
    
    if args.save_model:
        torch.save(model, args.model_path)
    if load_model == True:
        model = torch.load(args.model_path)

    if args.calc_map == False:
        ap = NewAveragePrecision(model, test, k=20)
        print("map: ", ap.mean())

    predict(args, model, df_rate, neg_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_data", action="store_true")
    parser.add_argument("--mode", type=str, default="original", help="original or mine")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--calc_map", action="store_true", help="whether to calculate map every epoch")
    parser.add_argument("--neg_mode", type=str, default="original", help="original or mine")
    parser.add_argument("--num_negative_samples", type=int, default=50)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_path", type=str, default="model.h5")
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--representation", type=str)
    parser.add_argument("--min_seq_len", type=int, default=None)
    args = parser.parse_args()
    main(args)
