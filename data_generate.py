import matplotlib as mpl
mpl.use('Agg')

from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from group_norm import GroupNormalization
import random
import pandas as pd
import numpy as np
from Bio import SeqIO

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def dataProcessing(seq, key):
    #################### 2222222222222222222222222222222  ############################
    bases2 = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    X_2 = np.zeros((len(seq), len(seq[0]), 16))
    for l, s in enumerate(seq):
        s2=s+s[0]
        res = list(zip(s2, s2[1:]))
        for i, char in enumerate(res):
            char = [i[0] for i in char][0] + [i[0] for i in char][1]
            if char in bases2:
                X_2[l, i, bases2.index(char)] = 1
            else:
                print('NO')
    #################### 2222222222222222222222222222222  ############################
    bases = ['A', 'C', 'G', 'T']
    X = np.zeros((len(seq), len(seq[0]), len(bases)))
    for l, s in enumerate(seq):
        for i, char in enumerate(s):
            if char in bases:
                X[l, i, bases.index(char)] = 1
    chem_bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0, ], 'T': [0, 0, 1]}
    Z = np.zeros((len(seq), len(seq[0]), 3))
    for l, s in enumerate(seq):
        for i, char in enumerate(s):
            if char in chem_bases:
                Z[l][i] = (chem_bases[char])

    all_features = np.concatenate([X, Z,X_2], axis=2)
    if key == 1:
        lbs = list(np.ones(len(X)))
    if key == 2:
        lbs = list(np.zeros(len(X)))
    y = np.array(lbs, dtype=np.int32)

    return all_features, y

def dataProcessing2(seq, key):
    #################### 2222222222222222222222222222222  ############################
    bases2 = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    X_2 = np.zeros((len(seq), len(seq[0]), 16))
    for l, s in enumerate(seq):
        res = list(zip(s, s[1:]))
        for i, char in enumerate(res):
            char = [i[0] for i in char][0] + [i[0] for i in char][1]
            if char in bases2:
                X_2[l, i, bases2.index(char)] = 1
    #################### 2222222222222222222222222222222  ############################
    all_features = X_2
    if key == 1:
        lbs = list(np.ones(len(X_2)))
    if key == 2:
        lbs = list(np.zeros(len(X_2)))
    y = np.array(lbs, dtype=np.int32)

    return all_features, y

def prepareData(path):
    pos_train=path+'train_pos.fa'
    pos_test = path + 'test_pos.fa'
    neg_train=path+'train_neg.fa'
    neg_test = path + 'test_neg.fa'
    pos_seq = []
    neg_seq=[]
    for seq_record in SeqIO.parse(pos_train,"fasta"):
        pos_seq.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(pos_test,"fasta"):
        pos_seq.append(str(seq_record.seq))

    for seq_record in SeqIO.parse(neg_train,"fasta"):
        neg_seq.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(neg_test,"fasta"):
        neg_seq.append(str(seq_record.seq))

    # for i in range(len(all_seq)):
    #     if (i < (len(all_seq) / 2)):
    #         pos_seq.append(all_seq[i])
    #     else:
    #         neg_seq.append(all_seq[i])
    a = 1
    b = 2

    Positive_X, Positive_y = dataProcessing(pos_seq, a);
    Negitive_X, Negitive_y = dataProcessing(neg_seq, b);

    return Positive_X, Positive_y, Negitive_X, Negitive_y

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;

def test_data_prepro(ind_test):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(ind_test)
    test_X = np.concatenate((Positive_X, Negitive_X))
    test_y = np.concatenate((Positive_y, Negitive_y))
    return test_X, test_y

def funciton(All_data, folds):
    if not  os.path.exists('folds'):
        os.mkdir('folds')
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(All_data)

    random.shuffle(Positive_X);
    random.shuffle(Negitive_X);

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))

        validation_index = (test_index + 1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index], Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index], Negative_y_Slices[validation_index]))

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X, test_y);
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y);
        np.save('folds/' + str(test_index) + '_' + 'x_test', test_X)
        np.save('folds/' + str(test_index) + '_' + 'y_test', test_y)
        np.save('folds/' + str(test_index) + '_' + 'valid_X', valid_X)
        np.save('folds/' + str(test_index) + '_' + 'valid_y', valid_y)
        np.save('folds/' + str(test_index) + '_' + 'x_train', train_X)
        np.save('folds/' + str(test_index) + '_' + 'y_train', train_y)
import os
All_data='/home/isrldl/hdd_4T/Naeem/Bioinformatics/Final paper codes/data/hs/'

OutputDir = 'output/'
funciton(All_data, 5);