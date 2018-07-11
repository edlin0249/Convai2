
idx_PAD = 0
idx_START = 1
idx_EOS = 2
idx_UNK = 3
# store parsed data, vocab and embedding weight
def word2idx(sentence, vocab2index, index2vocab):
    ans = []
    for w in sentence.split(' '):
        if w == "":
            continue
        ans.append(vocab2index[w] if w in vocab2index else idx_UNK)
    ans.append(idx_EOS)
    return ans

persona_begins = ["partner_s_persona: ", "your_persona: "]
def dialog2dict(d, vocab2index, index2vocab):
    tmp = {'partner':[], 'me':[], 'dialog':[], 'candidates':[], 'ans_ids':[]} # 'dialog'[dialog_len][2] 0: partner's sentence, 1: my sentence
    for line in d.strip().split('\n'):
        line_dict = line.split(" ", 1)[1].strip().split("\t")
        if len(line_dict) == 1: # persona
            if persona_begins[0] == line_dict[0][0:len(persona_begins[0])]:
                tmp['partner'].append( word2idx(line_dict[0][len(persona_begins[0]):], vocab2index, index2vocab) )
            elif persona_begins[1] == line_dict[0][0:len(persona_begins[1])]:
                tmp['me'].append( word2idx(line_dict[0][len(persona_begins[1]):], vocab2index, index2vocab) )
            else:
                assert(False) # must be persona
        elif len(line_dict) == 2: # dialog
            tmp['dialog'].append(list(map(lambda x: word2idx(x, vocab2index, index2vocab), line_dict)))
        else: # dialog with cadidates
            cands = line_dict[3].split('|')
            line_dict = line_dict[:2]
            for i, s in enumerate(cands):
                if s == line_dict[1]:
                    tmp['ans_ids'].append(i)
            tmp['dialog'].append(list(map(lambda x: word2idx(x, vocab2index, index2vocab), line_dict)))
            tmp['candidates'].append(list(map(lambda x: word2idx(x, vocab2index, index2vocab), cands)))
    return tmp

def load_data(data_file, vocab2index=None, index2vocab=None):
    f = open(data_file)
    context = f.read()
    f.close()
    text = context.replace("partner's persona: ", "").replace("your persona: ", "").replace("'", " '").replace("\n", " ").replace("\t", " ").replace(".", " .").replace("|"," ").strip().split(" ") # i'll -> i 'll
    if vocab2index==None:
        vocab2index = {}
        for w in text:
            if w == "":
                continue
            if w not in vocab2index:
                vocab2index[w] = 1
            else:
                vocab2index[w] += 1
        index2vocab = ["<PAD>", "<START>", "<EOS>", "<UNK>"]
        # 0:pad 1:start 2:end 3:unknown
        for i, w in enumerate(vocab2index):
            if vocab2index[w] >= 0:  # trashold for word frequency
                vocab2index[w] = len(index2vocab)
                index2vocab.append(w)
        
    dialogs = context.replace("partner's persona: ", "partner_s_persona: ").replace("your persona: ", "your_persona: ").replace(".", " .").replace("'", " '").replace("n 't", " n't").split("\n1 ")
    dialogs[0] = dialogs[0]+"\n"
    for i in range(1, len(dialogs)-1):
        dialogs[i] = "1 " + dialogs[i] + "\n"
    dialogs[-1] = "1 " + dialogs[-1]
    # index 0: partner, 1: myself(our model)
    # while True:
    #     index = int(input(":"))
    #     print(dialogs[index])
    #     for i,j in epsodes[index].items():
    #         print(i, j)
    #         print("------")
    #     print("\n\n\n")
    # symbol = "__SILENCE__"
    return [ dialog2dict(d, vocab2index, index2vocab) for d in dialogs ], vocab2index, index2vocab

def load_embedding(embedding_file, vocab2index, index2vocab):
    embedding_weight = [ "x" for i in index2vocab ]
    f = open(embedding_file)
    embedding_dim = 100
    for line in f:
        w, vec_str = line.strip().split(" ", 1)
        if w in vocab2index:
            tmp = [ float(s) for s in vec_str.split(' ') ]
            embedding_weight[vocab2index[w]] = tmp
            assert(len(tmp) == embedding_dim)
    for i, x in enumerate(embedding_weight):
        if x == "x":
            embedding_weight[i] = [0.0]*embedding_dim
            #print(index2vocab[i])
    return embedding_weight, embedding_dim

def get_persona_batch(eps):
    personas = [ d['me'] for d in eps ]
    # persona: batch_size, num_sentence, num_words
    max_num_sentence = 0
    max_num_words = 0
    for persona in personas:
        for sentence in persona:
            if len(sentence) > max_num_words:
                max_num_words = len(sentence)
        length = len(persona)
        if len(persona) > max_num_sentence:
            max_num_sentence = len(persona)

    for i in range(max_num_sentence):
        for persona in personas:
            if i < len(persona):
                persona[i] += [idx_PAD] * (max_num_words - len(persona[i]))
            else:
                persona.append([idx_PAD] * max_num_words)
    return personas

def get_turn_batch(dialog_batch, ith_turn, person_id):
    batch_s = []
    max_s_len = 0
    origin_num = 0
    for d in dialog_batch:
        if ith_turn < len(d):
            batch_s.append(d[ith_turn][person_id])
            if len(d[ith_turn][person_id]) > max_s_len:
                max_s_len = len(d[ith_turn][person_id])
        else:
            batch_s.append([])
    for s in batch_s:
        origin_num += len(s)
        s += [idx_PAD] * (max_s_len - len(s))
    return batch_s, origin_num
