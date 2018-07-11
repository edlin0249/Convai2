from parse_data import load_data, idx_PAD, get_persona_batch
from torch.autograd import Variable
import torch.optim as optim
import pickle
import torch
import seq2seq_model
from tqdm import tqdm
import argparse

def get_choice_batch(choices):
    max_s_len = 0
    for s in choices:
        if len(s) > max_s_len:
            max_s_len = len(s)
    len_list = []
    for s in choices:
        len_list.append(len(s))
        s += [0] * (max_s_len - len(s))
    return choices, len_list

def test_multi_choice(epsodes, model, criterion):
    model.eval()
    running_loss = 0.0
    total_num = 0
    correct_num = 0
    choice_num = 0
    for eps in tqdm(epsodes):
        cand_size = len(eps['candidates'][0])
        personas = get_persona_batch([eps]*cand_size)
        personas = Variable(torch.LongTensor(personas)).cuda()
        model.init_persona(personas)
        choice_num += len(eps['dialog'])
        for i in range(len(eps['dialog'])):
            batch_xs = Variable(torch.LongTensor([eps['dialog'][i][0]])).cuda().expand(cand_size, -1)
            batch_ys, len_list = get_choice_batch(eps['candidates'][i])
            batch_ys = Variable(torch.LongTensor(batch_ys)).cuda()
            pred, out = model(batch_xs, batch_ys)
            min_idx = 0
            loss = []
            for j in range(cand_size):
                tmp = criterion(out[j], batch_ys[j].view(-1)).data
                loss.append(tmp[0])
                if loss[j] <= loss[min_idx]:
                    min_idx = j
            running_loss += loss[eps['ans_ids'][i]]
            total_num += len_list[eps['ans_ids'][i]]
            if loss[eps['ans_ids'][i]] <= loss[min_idx]:
                correct_num += 1
            model.select_history(eps['ans_ids'][i]) # set considered-right history, clean other candidates
    return running_loss/total_num, correct_num/choice_num

def main(args):
    with open(args.vocab, 'rb') as f:
        vocab_dict = pickle.load(f)
    vocab2index = vocab_dict['vocab2index']
    index2vocab = vocab_dict['index2vocab']
    epsodes = load_data(args.data, vocab2index, index2vocab)[0]

    embedding_dim = 100
    model = seq2seq_model.Seq2Seq({'num_embeddings':len(index2vocab), 'embedding_dim':embedding_dim, 'embdding_weight':None,
                                   "rnn_class":torch.nn.GRU, 'hidden_size':128, 'num_layers':2, 'dropout':0.5, 'bidirectional':True,
                                   "history_size": 256*2, 'persona_size': embedding_dim}).cuda()
    model.load_state_dict(torch.load(args.model))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=idx_PAD, size_average=False).cuda()
    loss, choice_accu = test_multi_choice(epsodes[-100:], model, criterion)
    print(loss, choice_accu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="valid_both_original.txt")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vocab', type=str, default="vocab.pickle")

    main( parser.parse_args() )
