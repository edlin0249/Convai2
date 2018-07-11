# 1. build vocab
# 2. function sentence2index()
persona_begins = ["partner_s_persona: ", "your_persona: "]
f = open("train_both_original_no_cands.txt")
context = f.read()
f.close()
text = context.replace("partner's persona: ", "").replace("your persona: ", "").replace("'", " '").replace("\n", " ").replace("\t", " ").replace(".", " .").strip().split(" ") # i'll -> i 'll
vocab2index = {}
for w in text:
    if w == "":
        continue
    if w not in vocab2index:
        vocab2index[w] = 1
    else:
        vocab2index[w] += 1
index2vocab = [None, None, None, None]
# 0:pad 1:start 2:end 3:unknown
idx_PAD = 0
idx_START = 1
idx_EOS = 2
idx_UNK = 3
for i, w in enumerate(vocab2index):
    if vocab2index[w] >= 0:  # trashold for word frequency
        vocab2index[w] = len(index2vocab)
        index2vocab.append(w)
    
dialogs = context.replace("partner's persona: ", "partner_s_persona: ").replace("your persona: ", "your_persona: ").replace(".", " .").replace("'", " '").replace("n 't", " n't").split("\n1 ")
dialogs[0] = dialogs[0]+"\n"
for i in range(1, len(dialogs)-1):
    dialogs[i] = "1 " + dialogs[i] + "\n"
dialogs[-1] = "1 " + dialogs[-1]

def word2idx(sentence):
    ans = []
    for w in sentence.split(' '):
        if w == "":
            continue
        ans.append(vocab2index[w] if w in vocab2index else 3)
    ans.append(idx_EOS)
    return ans
def dialog2dict(d):
    tmp = {'partner':[], 'me':[], 'dialog':[], 'option':None} # 'dialog'[dialog_len][2] 0: partner's sentence, 1: my sentence
    for line in d.strip().split('\n'):
        line_dict = line.split(" ", 1)[1].strip().split("\t")
        if len(line_dict) == 1: # persona
            if persona_begins[0] == line_dict[0][0:len(persona_begins[0])]:
                tmp['partner'].append( word2idx(line_dict[0][len(persona_begins[0]):]) )
            elif persona_begins[1] == line_dict[0][0:len(persona_begins[1])]:
                tmp['me'].append( word2idx(line_dict[0][len(persona_begins[1]):]) )
            else:
                assert(False) # must be persona
        else: # dialog
            tmp['dialog'].append(list(map(word2idx, line_dict)))
    return tmp
epsodes = [ dialog2dict(d) for d in dialogs ]
# index 0: partner, 1: myself(our model)
# while True:
#     index = int(input(":"))
#     print(dialogs[index])
#     for i,j in epsodes[index].items():
#         print(i, j)
#         print("------")
#     print("\n\n\n")
# symbol = "__SILENCE__"

embedding_weight = [ "x" for i in index2vocab ]
f = open("glove.6B.100d.txt")
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

batch_size = 32
save_round = 3
#epsodes
import torch
from torch.autograd import Variable
import seq2seq_model
import torch.optim as optim
import datetime
import os
import codecs
from tqdm import tqdm

date = datetime.datetime.now().strftime("%d-%H-%M")
save_path = 'model/model_{}'.format(date)
print ('save_path = {}'.format(save_path))
if not os.path.exists(save_path):
    os.mkdir(save_path)
log_file = codecs.open(os.path.join(save_path, 'log'), 'w')
embedding_weight = torch.Tensor(embedding_weight)
model = seq2seq_model.Seq2Seq({'num_embeddings':len(index2vocab), 'embedding_dim':embedding_dim, 'embdding_weight':embedding_weight,
                               "rnn_class":torch.nn.GRU, 'hidden_size':128, 'num_layers':2, 'dropout':0.5, 'bidirectional':True,
                               "history_size": 256*2, 'persona_size': embedding_dim}).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters())
for e in range(300):
    model.train()
    running_loss = 0
    for b_start in range(0, len(epsodes) - batch_size + 1, batch_size):
        personas = []
        dialog_batch = []
        for i in range(batch_size):
            personas.append(epsodes[b_start+i]['me'])
            dialog_batch.append(epsodes[b_start+i]['dialog'])
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
                    persona[i] += [0] * (max_num_words - len(persona[i]))
                else:
                    persona.append([0] * max_num_words)

        personas = Variable(torch.LongTensor(personas)).cuda()
        model.init_persona(personas)
        #print(b_start)

        max_num_turn = max([ len(d) for d in dialog_batch ])
        for i in range(max_num_turn):
            batch_xs = []
            batch_ys = []
            max_xs_len = 0
            max_ys_len = 0
            for d in dialog_batch:
                if i < len(d):
                    batch_xs.append(d[i][0])
                    batch_ys.append(d[i][1])
                    if len(d[i][0]) > max_xs_len:
                        max_xs_len = len(d[i][0])
                    if len(d[i][1]) > max_ys_len:
                        max_ys_len = len(d[i][1])
                else:
                    batch_xs.append([])
                    batch_ys.append([])
            for x in batch_xs:
                x += [0] * (max_xs_len - len(x))
            batch_xs = Variable(torch.LongTensor(batch_xs)).cuda()
            for y in batch_ys:
                y += [0] * (max_ys_len - len(y))
            batch_ys = Variable(torch.LongTensor(batch_ys)).cuda()
            optimizer.zero_grad()
            pred, out = model(batch_xs, batch_ys)
            #print ('model finish')
            out = out.view(-1, out.size(2))
            #print (out.size(), batch_ys.size())
            loss = criterion(out.cuda(), batch_ys.view(-1).cuda())
    #        torch.cuda.synchronize()
            loss.backward(retain_graph=True)
            optimizer.step()
            print ('epsodes = {}, b_start = {}, i = {}, loss = {}'.format(e, b_start, i, loss))
            running_loss += loss.data
            del batch_xs, batch_ys, pred, out
            #loss(out, xs)
    print ('epsodes = {}, running_loss = {}'.format(e, running_loss))
    print ('epsodes = {}, running_loss = {}'.format(e, running_loss), log_file)
    if e % save_round == save_round - 1:
        with open(os.path.join(save_path, 'model_{}'.format(e)), 'wb') as f:
                torch.save(model.state_dict(), f)
