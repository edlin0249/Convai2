import numpy as np
import torch

# person_id = 0 , the other person
#           = 1 , myself
# eps = { 'persona':([],[]), 'dialog':([],[]), 'candidates':[], 'ans_ids':[] }
# eps['persona'][person_id][ith_sentence]: persona for model is eps['persona'[1]
# eps['dialog'][person_id][ith_sentence]: when the i-th turn, x = eps['dialog'][0][ith_turn], y = eps['dialog'][1][ith_turn]

idx_PAD = 0
idx_START = 1
idx_EOS = 2
idx_UNK = 3
idx_BOC = 5
idx_EOC = 6

persona_begins = ["partner 's persona: ", "your persona: "]
def split_data(data_file):
    f = open(data_file)
    context = f.read()
    f.close()
    lines = context.replace(".", " .").replace("'", " '").replace("n 't", " n't").strip().split('\n')
    episodes = []
    eps = { 'persona':([],[]), 'dialog':([]), 'candidates':[], 'ans_ids':[] }
    for line_number, line in enumerate(lines):
        line = line.split(" ", maxsplit=1)[1] # remove the number at the beginning of line
        line_list = line.split('\t')
        if len(line_list) == 1:
            finded = False
            for i, s in enumerate(persona_begins):
                if line.startswith(s):
                    eps['persona'][i].append(line[len(s):])
                    finded = True #for debug
                    break
            assert(finded) #for debug
        else:
            eps['dialog'].append("<START> "+line_list[0]+" <EOS>")
            eps['dialog'].append("<START> "+line_list[1]+" <EOS>")
            if len(line_list) == 4:
                cand_list = line_list[3].split("|")
                eps['candidates'].append(cand_list)
                finded = False
                for i, s in enumerate(cand_list):
                    if s == line_list[1]:
                        eps['ans_ids'].append(i)
                        finded = True #for debug
                        break
                assert(finded) #for debug
        if line_number == len(lines)-1 or lines[line_number+1].startswith("1 "):
            eps['dialog'][0] = "<BOC> " + eps['dialog'][0]
            eps['dialog'][-1] = eps['dialog'][-1] + " <EOC>"
            #print(eps['dialog'][0])
            #print(eps['dialog'][-1])
            episodes.append(eps)
            #print(eps)
            eps = { 'persona':([],[]), 'dialog':([]), 'candidates':[], 'ans_ids':[] }
    return episodes

def count_vocab(sentence_list, count_dict):
    assert(type(sentence_list) == list)
    for s in sentence_list:
        for w in s.split(' '):
            if w == "":
                continue
            if w not in count_dict:
                count_dict[w] = 1
            else:
                count_dict[w] += 1
            
def build_vocab(episodes, embedding_file=None, dim_in_file=None, train_oov=True, count_threshold=0):
    count_dict = {}
    for eps in episodes:
        count_vocab(eps['persona'][0], count_dict)
        count_vocab(eps['persona'][1], count_dict)
        count_vocab(eps['dialog'], count_dict)
        #count_vocab(eps['dialog'][1], count_dict)
    # 0:pad 1:start 2:end 3:unknown
    assert("" not in count_dict)
    for i, w in enumerate(count_dict):
        if count_dict[w] < count_threshold:  # trashold for word frequency
            del count_dict[w]

    index2vocab = ["<PAD>", "<START>", "<EOS>", "<UNK>", "__SILENCE__", "<BOC>", "<EOC>"]
    if embedding_file == None:
        index2vocab = index2vocab + [ w for w in count_dict ]
        embedding_weight, embedding_dim = None, Nonen
    else:
        embedding_weight = [ [0]*dim_in_file for w in index2vocab ]
        f = open(embedding_file)
        for line in f:
            w, vec_str = line.strip().split(" ", 1)
            if w in count_dict:
                tmp = [ float(s) for s in vec_str.split(' ') ]
                assert(len(tmp) == dim_in_file) # for debug
                embedding_weight.append(tmp)
                index2vocab.append(w)
                del count_dict[w]
        f.close()
        if train_oov:
            for w in count_dict:
                embedding_weight.append([0]*dim_in_file)
                index2vocab.append(w)
            embedding_weight = np.array(embedding_weight)
            embedding_dim = dim_in_file
        else:
            embedding_weight = np.array(embedding_weight)
            embedding_mean = np.sum(embedding_weight, axis=0) / (len(index2vocab)-5)
            embedding_weight[idx_UNK, :] = embedding_mean  # <UNK> embedding to be others average.
            one_hot_part = np.zeros((len(index2vocab), 4))
            one_hot_part[0][0] = one_hot_part[1][1] = one_hot_part[2][2] = one_hot_part[4][3] = 1
            embedding_weight = np.hstack((embedding_weight, one_hot_part))
            embedding_dim = dim_in_file + 4
    vocab2index = { w:i for i, w in enumerate(index2vocab) }
    return vocab2index, index2vocab, embedding_weight, embedding_dim

def word2idx(sentence, vocab2index):
    ans = []
    for w in sentence.split(' '):
        if w == "":
            continue
        ans.append(vocab2index[w] if w in vocab2index else idx_UNK)
    #ans.append(idx_EOS)
    return ans

def sentence_list2index(sentence_list, vocab2index):
    assert(type(sentence_list) == list)
    for i, s in enumerate(sentence_list):
        sentence_list[i] = word2idx(s, vocab2index)

def episodes_text2index(episodes, vocab2index):
    for eps in episodes:
        sentence_list2index(eps['persona'][0], vocab2index)
        sentence_list2index(eps['persona'][1], vocab2index)
        sentence_list2index(eps['dialog'], vocab2index)
        #sentence_list2index(eps['dialog'][1], vocab2index)
        for candidates_list in eps['candidates']:
            sentence_list2index(candidates_list, vocab2index)

def get_persona_batch(episodes, idx_01):
    personas = [ d['persona'][idx_01] for d in episodes ]
    # personas: batch_size, num_sentence, num_words
    max_num_sentence = 0
    max_num_words = 0
    for persona in personas:
        for sentence in persona:
            if len(sentence) > max_num_words:
                max_num_words = len(sentence)
        length = len(persona)
        if len(persona) > max_num_sentence:
            max_num_sentence = len(persona)
    ret = torch.LongTensor(len(episodes), max_num_sentence, max_num_words).cuda()
    ret.fill_(idx_PAD)
    for i, p in enumerate(personas):
        for j, sentence in enumerate(p):
            for k, word_index in enumerate(sentence):
                ret[i, j, k] = word_index
    return ret

def get_turn_batch(episodes, ith_turn):
    pass
"""
    max_num_words = 0
    num = 0
    for eps in episodes:
        sentence_list = eps['dialog']
        if ith_sent < len(sentence_list) and len(sentence_list[ith_sent]) > max_num_words:
            max_num_words = len(sentence_list[ith_turn])
    ret = torch.LongTensor(len(episodes), max_num_words)
    ret.fill_(idx_PAD)
    for i, eps in enumerate(episodes):
        sentence_list = eps['dialog']
        if ith_sent < len(sentence_list):
            num += len(sentence_list)
            for j, word_index in enumerate(sentence_list[ith_turn]):
                ret[i, j] = word_index
    return ret, num
"""

def get_dialog_batches(episodes):

    max_num_words = 0
    for idx1, eps in enumerate(episodes):
        tmp = []
        for sent in eps['dialog']:
            tmp += sent
        if len(tmp) > max_num_words:
            max_num_words = len(tmp)
        del tmp[:]

    dialog_batches = []

    for idx1, eps in enumerate(episodes):
        tmp = []
        for sent in eps['dialog']:
            tmp += sent
        tmp += [idx_PAD]*(max_num_words - len(tmp))
        dialog_batches.append(tmp)
    dialog_batches = torch.LongTensor(dialog_batches).cuda()

    return dialog_batches