from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

from parse_data_ver4 import *
import generator
import discriminator
import helpers
from tqdm import tqdm

import random

CUDA = torch.cuda.is_available()
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 300
#START_LETTER = 0
#BATCH_SIZE = 20
MLE_TRAIN_EPOCHS = 1
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

idx_PAD = 0
idx_START = 1
idx_EOS = 2
idx_UNK = 3
idx_BOC = 5
idx_EOC = 6

def train_generator_MLE(gen, gen_opt, episodes, valid_episodes, batch_size, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    #print("batch_size = %d"%batch_size)
    #print("epochs = %d"%epochs)
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1))
        sys.stdout.flush()
        total_loss = 0
        total_size = 0
        
        print("train:")
        print("len(episodes) = %d"%len(episodes))
        for batch_idx in range(0, len(episodes), batch_size):
            #print("batch_idx = %d"%batch_idx)
        #for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            #print("1")
            personas_your = get_persona_batch(episodes[batch_idx:batch_idx+batch_size], 1)
            #print("2")
            personas_partner = get_persona_batch(episodes[batch_idx:batch_idx+batch_size], 0)
            #print("3")
            turn_batch_list = get_dialog_batches(episodes[batch_idx:batch_idx+batch_size])
            #print("4")

            inp, target = helpers.prepare_generator_batch(turn_batch_list, gpu=CUDA)
            #print("5")
            gen_opt.zero_grad()
            #print("6")
            loss = gen.batchNLLLoss(inp, target, personas_your, personas_partner)
            #print("7")
            loss.backward()
            gen_opt.step()

            print("epoch: %d, batch_idx: %d, loss per sample = %f"%(epoch+1, batch_idx, loss.data[0]/turn_batch_list.size(0)/turn_batch_list.size(1)))

            #total_loss += loss.data[0]
            #total_size += turn_batch_list.size(0)

            #if (i / BATCH_SIZE) % ceil(
                            #ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                #print('.', end='')
                #sys.stdout.flush()
            #if idx % 10 == 0:
                #print('.', end='')
                #sys.stdout.flush()

        if epoch % 5 == 0:
            print("valid:")
            for batch_idx in range(0, len(valid_episodes), batch_size):
            #for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
                personas_your = get_persona_batch(valid_episodes[batch_idx:batch_idx+batch_size], 1)
                personas_partner = get_persona_batch(valid_episodes[batch_idx:batch_idx+batch_size], 0)
                turn_batch_list = get_dialog_batches(valid_episodes[batch_idx:batch_idx+batch_size])
                inp, target = helpers.prepare_generator_batch(turn_batch_list, gpu=CUDA)
                #gen_opt.zero_grad()
                loss = gen.batchNLLLoss(inp, target, personas_your, personas_partner)
                #loss.backward()
                #gen_opt.step()

                print("epoch: %d, batch_idx: %d, loss per sample = %f"%(epoch+1, batch_idx, loss.data[0]/turn_batch_list.size(0)/turn_batch_list.size(1)))

        # each loss in a batch is loss per sample
        #total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        #oracle_loss = helpers.batchwise_oracle_nll(gen, valid_batch_list, BATCH_SIZE, MAX_SEQ_LEN, gpu=CUDA)

        #print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(gen, gen_opt, dis, batch_size, episodes, num_batches, Sample_Size=20):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        random_sample_index = np.random.choice(len(episodes), Sample_Size, replace=False)
        random_episodes = episodes[random_sample_index]
        s, condition = gen.sample(random_episodes, idx_BOC)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, gpu=CUDA)
        rewards = dis.batchClassify(target, condition)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards, condition)
        pg_loss.backward()
        gen_opt.step()
        print("PG Loss = %f"%pg_loss.data[0])

    # sample from generator and compute oracle NLL
    #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                  # start_letter=START_LETTER, gpu=CUDA)

    #print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(dis, dis_opt, episodes, valid_episodes, gen, batch_size, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    
    #pos_val = oracle.sample(100)
    #neg_val = generator.sample(100)
    #val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)
    
    for d_step in range(d_steps):
        print("d-step = %d"%(d_step+1))
        #s = helpers.batchwise_sample(generator, episodes, batch_size)
        #s, condition = gen.sample(episodes, idx_BOC)
        #dis_inp, dis_target = helpers.prepare_discriminator_data(episodes, s, condition, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for batch_idx in range(0, len(episodes), batch_size):
                s, condition = gen.sample(episodes[batch_idx:batch_idx+batch_size], idx_BOC)
                #inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                loss_fn = nn.BCELoss()
                #True case == target is one
                inp = get_dialog_batches(episodes[batch_idx:batch_idx+batch_size])
                out = dis.batchClassify(inp, condition)
                target = torch.ones(inp.size(0)).cuda()
                True_loss = loss_fn(out, target)
                True_loss.backward(retain_graph=True)
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #False case == target is zero
                #inp = s
                out = dis.batchClassify(s, condition)
                target = torch.zeros(inp.size(0)).cuda()
                False_loss_fake_img = loss_fn(out, target)
                False_loss_fake_img.backward(retain_graph=True)
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #False case == target is zero
                random_index = np.random.choice(condition.size(0), condition.size(0), replace=False)
                random_condition = condition[random_index]
                inp = get_dialog_batches(episodes[batch_idx:batch_idx+batch_size])
                out = dis.batchClassify(inp, random_condition)
                target = torch.zeros(inp.size(0)).cuda()
                False_loss_notmatch = loss_fn(out, target)
                False_loss_notmatch.backward(retain_graph=True)
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #loss = True_loss + False_loss_fake_img + False_loss_notmatch

                #loss.backward()
                dis_opt.step()

                total_loss += True_loss.data[0]+False_loss_fake_img.data[0]+False_loss_notmatch.data[0]
                #total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]

                if (batch_idx % 10) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(3 * len(episodes) / float(batch_size))
            total_acc /= float(3 * len(episodes))

            print("total_loss = %f"%total_loss)
            print("total_acc = %f"%total_acc)

        if epoch % 5 == 0:
            val_total_loss = 0
            val_total_acc = 0
            #s, condition = gen.sample(generator, valid_episodes, idx_START)
            for batch_idx in range(0, len(valid_episodes), batch_size):
                s, condition = gen.sample(valid_episodes[batch_idx:batch_idx+batch_size], idx_BOC)
                #inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                #dis_opt.zero_grad()
                loss_fn = nn.BCELoss()
                #True case == target is one
                val_inp = get_dialog_batches(valid_episodes[batch_idx:batch_idx+batch_size])
                out = dis.batchClassify(val_inp, condition)
                target = torch.ones(val_inp.size(0)).cuda()
                True_loss = loss_fn(out, target)
                #True_loss.backward()
                val_total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #False case == target is zero
                #val_inp = s[batch_idx:batch_idx+batch_size]
                out = dis.batchClassify(s, condition)
                target = torch.zeros(s.size(0)).cuda()
                False_loss_fake_img = loss_fn(out, target)
                #False_loss_fake_img.backward()
                val_total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #False case == target is zero
                random_index = np.random.choice(condition.size(0), condition.size(0), replace=False)
                random_condition = condition[random_index]
                val_inp = get_dialog_batches(valid_episodes[batch_idx:batch_idx+batch_size])
                out = dis.batchClassify(val_inp, random_condition)
                target = torch.zeros(val_inp.size(0)).cuda()
                False_loss_notmatch = loss_fn(out, target)
                #False_loss_notmatch.backward()
                val_total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                #loss = True_loss + False_loss_fake_img + False_loss_notmatch

                #loss.backward()
                #dis_opt.step()

                val_total_loss += True_loss+False_loss_fake_img+False_loss_notmatch
                #total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]

                if (batch_idx % 10) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            val_total_loss /= ceil(3 * len(valid_episodes) / float(batch_size))
            val_total_acc /= float(3 * len(valid_episodes))

            print("val_total_loss = %f"%val_total_loss)
            print("val_total_acc = %f"%val_total_acc)

            #val_pred = discriminator.batchClassify(val_inp)
            #print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
            #    total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data[0]/200.))