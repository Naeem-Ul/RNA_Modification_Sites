import matplotlib as mpl
mpl.use('Agg')

from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,AveragePooling1D,concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from group_norm import GroupNormalization
import random
import pandas as pd
import numpy as np
from Bio import SeqIO
from keras import regularizers
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os, sys, copy, getopt, re, argparse
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras import losses
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from scipy import interp
from sklearn.utils.class_weight import compute_class_weight


def analyze(temp, OutputDir):
    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title + 'results\n')

        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:  # ,'pre_recall_curve'

            total = []

            for val in x:
                total.append(val[j])
            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total)) + '\n')

        file.write('\n\n______________________________\n')
    file.close();

    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig(OutputDir + '/' + title + 'ROC.png')
        plt.close('all');

        # ************************** Precision Recall Curve*********************************
        i = 0
        prs = []
        pre_aucs = []
        mean_recal = np.linspace(0, 1, 100)
        for val in x:
            pre = val['prec']
            rec = val['reca']
            prs.append(interp(mean_recal, rec, pre))
            prs[-1][0] = 0.0
            p_r_auc = auc(rec, pre)
            pre_aucs.append(p_r_auc)
            plt.plot(rec, pre, lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.2f)' % (i + 1, p_r_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

        mean_pre = np.mean(prs, axis=0)
        mean_pre[-1] = 1.0
        mean_auc = auc(mean_recal, mean_pre)
        std_auc = np.std(pre_aucs)
        plt.plot(mean_recal, mean_pre, color='b',
                 label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_pre = np.std(prs, axis=0)
        pre_upper = np.minimum(mean_pre + std_pre, 1)
        pre_lower = np.maximum(mean_pre - std_pre, 0)
        plt.fill_between(mean_recal, pre_lower, pre_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'
        if index == 3:
            title = 'ind_testing_'

        plt.savefig(OutputDir + '/' + title + 'Pre_R_C.png')
        plt.close('all');

        index += 1;




def calculate(sequence):
    X = []
    dictNum = {'A': 0, 'U': 0, 'C': 0, 'G': 0};

    for i in range(len(sequence)):

        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));

    return np.array(X)


def getMode():
    input_shape1 = (41, 4)
    inputs1 = Input(shape=input_shape1)
    input_shape2 = (41, 16)
    inputs2 = Input(shape=input_shape2)
    input_shape3 = (41, 3)
    inputs3 = Input(shape=input_shape3)

    convLayer = Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                       bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape1)(inputs1);
    normalizationLayer = GroupNormalization(groups=4, axis=-1)(convLayer)
    convLayer2 = Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                        bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape1)( normalizationLayer)
    flattenLayer = Flatten()(convLayer2)
    dropoutLayer = Dropout(0.5)(flattenLayer)
    denseLayer = Dense(8*3, activation='relu', kernel_regularizer=regularizers.l2(1e-3),bias_regularizer=regularizers.l2(1e-4))(dropoutLayer)

    convLayer_b = Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                       bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape2)(inputs2);
    normalizationLayer_b = GroupNormalization(groups=4, axis=-1)(convLayer_b)
    convLayer2_b = Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                        bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape2)( normalizationLayer_b)
    flattenLayer_b = Flatten()(convLayer2_b)
    dropoutLayer_b = Dropout(0.5)(flattenLayer_b)
    denseLayer_b = Dense(8*3, activation='relu', kernel_regularizer=regularizers.l2(1e-3),bias_regularizer=regularizers.l2(1e-4))(dropoutLayer_b)

    convLayer_c = Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                       bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape3)(inputs3);
    normalizationLayer_c = GroupNormalization(groups=4, axis=-1)(convLayer_c)
    convLayer2_c = Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-3),
                        bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=input_shape3)( normalizationLayer_c)
    flattenLayer_c = Flatten()(convLayer2_c)
    dropoutLayer_c = Dropout(0.5)(flattenLayer_c)
    denseLayer_c = Dense(8*3, activation='relu', kernel_regularizer=regularizers.l2(1e-3),bias_regularizer=regularizers.l2(1e-4))(dropoutLayer_c)



    denseLayer_com=concatenate([denseLayer,denseLayer_c,denseLayer_b])
    outLayer = Dense(1, activation='sigmoid')(denseLayer_com)
    model = Model(inputs=[inputs1,inputs3,inputs2], outputs=outLayer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=[binary_accuracy]);  # SGD(lr = 0.005, momentum=0.95)
    print(model.summary())
    return model;


def calculateScore(X, y, model):
    score = model.evaluate([X[:,:,:4],X[:,:,4:7],X[:,:,7:]], y)
    pred_y = model.predict([X[:,:,:4],X[:,:,4:7],X[:,:,7:]])

    accuracy = score[1];

    tempLabel = np.zeros(shape=y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    pred_y = pred_y.reshape((-1,))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    pre, rec, threshlds = precision_recall_curve(y, pred_y)
    pre = np.fliplr([pre])[0]  # so the array is increasing (you won't get negative AUC)
    rec = np.fliplr([rec])[0]
    AUC_prec_rec = np.trapz(rec, pre)
    AUC_prec_rec = abs(AUC_prec_rec)

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)

    with tf.Session():
        lossValue = losses.binary_crossentropy(y_true, y_pred).eval()  # ,'pre_recall_curve':AUC_prec_rec

    return {'sn': sensitivity, 'sp': specificity, 'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea, 'precision': precision,
            'F1': F1Score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'lossValue': lossValue,
            'pre_recall_curve': AUC_prec_rec, 'prec': pre, 'reca': rec}

def Main(OutputDir, folds):

    trainning_result = []
    validation_result = []
    testing_result = []
    # ind_testing_result = []

    for test_index in range(folds):

        model = getMode();
        train_X, train_y,valid_X , valid_y,test_X,test_y = np.load('folds/'+str(test_index) +  '_x_train.npy'), np.load('folds/'+str(test_index) +  '_y_train.npy'),\
                                                           np.load('folds/' + str(test_index) + '_valid_X.npy'), np.load('folds/' + str(test_index) + '_valid_y.npy'),\
        np.load('folds/' + str(test_index) + '_x_test.npy'), np.load('folds/' + str(test_index) + '_y_test.npy'),

        result_folder = OutputDir
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        model_results_folder = result_folder

        # best_weights = model_results_folder + 'best_weights.h5'

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_check = ModelCheckpoint(filepath=OutputDir + "/model" + str(test_index + 1) + ".h5", mode='min',
                                      monitor='val_loss', save_best_only=True)  # , save_weights_only=True
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10)

        cbacks = [model_check, reduct_L_rate, early_stopping]

        history = model.fit([train_X[:,:,:4],train_X[:,:,4:7],train_X[:,:,7:]], train_y, batch_size=32, epochs=1000, validation_data=([valid_X[:,:,:4],valid_X[:,:,4:7],valid_X[:,:,7:]], valid_y),
                            callbacks=cbacks);

        trainning_result.append(calculateScore(train_X, train_y, model));
        validation_result.append(calculateScore(valid_X, valid_y, model));
        testing_result.append(calculateScore(test_X, test_y, model));
        # ind_testing_result.append(calculateScore(ind_test_X, ind_test_y, model));

    temp_dict = (trainning_result, validation_result, testing_result)
    analyze(temp_dict, OutputDir);

OutputDir = 'output/'
Main(OutputDir, 10);