import argparse
import torch
from torch.autograd import Variable
import seq2seq_model
import torch.optim as optim
import datetime
import os
import codecs
from tqdm import tqdm
import numpy as np
import pickle
from parse_data import *

def one_epoch(eth, episodes, model, criterion, optimizer, batch_size, train=True):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    total_num = 0
    for b_start in tqdm(range(0, len(episodes) - batch_size + 1, batch_size)):
        personas = get_persona_batch(episodes[b_start:b_start+batch_size])
        personas = Variable(torch.LongTensor(personas)).cuda()
        model.init_persona(personas)
        #print(b_start)

        dialog_batch = [ d['dialog'] for d in episodes[b_start:b_start+batch_size] ]
        max_num_turn = max([ len(d) for d in dialog_batch ])
        for i in range(max_num_turn):
            batch_xs = get_turn_batch(dialog_batch, i, 0)[0]
            batch_xs = Variable(torch.LongTensor(batch_xs)).cuda()
            batch_ys, num = get_turn_batch(dialog_batch, i, 1) # batch ys length differs, so scale of size average differs, need true num of words
            tmp = torch.LongTensor(batch_ys)
            # num = tmp.numpy().size
            batch_ys = Variable(tmp).cuda()
            if train:
                optimizer.zero_grad()
            pred, out = model(batch_xs, batch_ys)
            #print ('model finish')
            out = out.view(-1, out.size(2))
            #print (out.size(), batch_ys.size())
            loss = criterion(out.cuda(), batch_ys.view(-1).cuda())
            #torch.cuda.synchronize()
            #print ('episodes = {}, b_start = {}, i = {}, loss = {}'.format(eth, b_start, i, loss))
            if train:
                loss.backward(retain_graph=True)
                optimizer.step()
            running_loss += loss.data*num
            total_num += num
            del batch_xs, batch_ys, pred, out
            #loss(out, xs)
    return running_loss/total_num

def main(args):
    episodes = split_data(args.data)

    episodes = episodes[:len(episodes)//30] # for debug
    valid_rate = 0.15
    episodes = np.array(episodes, dtype=object)
    valid_num = int(valid_rate*len(episodes))
    valid_episodes = episodes[:valid_num]
    episodes = episodes[valid_num:]

    vocab2index, index2vocab = build_vocab(episodes, args.embedding, embedding_dim)
    embedding_weight, embedding_dim = load_embedding(args.embedding, vocab2index, index2vocab)

    episodes = episodes[:len(episodes)//30] # for debug
    valid_rate = 0.15
    episodes = np.array(episodes, dtype=object)
    valid_num = int(valid_rate*len(episodes))
    valid_episodes = episodes[:valid_num]
    episodes = episodes[valid_num:]

    batch_size = args.batch_size
    save_round = 1

    date = datetime.datetime.now().strftime("%d-%H-%M")
    save_path = 'model/model_{}'.format(date)
    print ('save_path = {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'vocab.pickle'), 'wb') as f:
        pickle.dump({'vocab2index':vocab2index, 'index2vocab':index2vocab}, f)
    log_file = codecs.open(os.path.join(save_path, 'log'), 'w')
    embedding_weight = torch.Tensor(embedding_weight)
    model = seq2seq_model.Seq2Seq({'num_embeddings':len(index2vocab), 'embedding_dim':embedding_dim, 'embdding_weight':embedding_weight,
                                   "rnn_class":torch.nn.GRU, 'hidden_size':128, 'num_layers':2, 'dropout':0.5, 'bidirectional':True,
                                   "history_size": 256*2, 'persona_size': embedding_dim}).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=idx_PAD).cuda()
    optimizer = optim.Adam(model.parameters())
    part_num = 2
    part_size = len(episodes)//part_num + 1
    for e in range(100):
        for p in range(part_num):
            loss = one_epoch(e, episodes[p*part_size:(p+1)*part_size], model, criterion, optimizer, batch_size, train=True)
            print ('episodes = {}, training_loss = {}'.format(e, loss))
            print ('episodes = {}, training_loss = {}'.format(e, loss), file=log_file)

            loss = one_epoch(e, valid_episodes, model, criterion, optimizer, batch_size, train=False)
            print ('episodes = {}, valid_loss = {}'.format(e, loss))
            print ('episodes = {}, valid_loss = {}'.format(e, loss), file=log_file)
        if e % save_round == save_round - 1:
            with open(os.path.join(save_path, 'model_{}'.format(e)), 'wb') as f:
                    torch.save(model.state_dict(), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default='32')
    parser.add_argument('--data', type=str, default="train_both_original_no_cands.txt")
    parser.add_argument('--embedding', type=str, default="glove.6B.100d.txt")

    main( parser.parse_args() )


