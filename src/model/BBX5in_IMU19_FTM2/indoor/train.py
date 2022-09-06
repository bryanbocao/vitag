from __future__ import division

'''
Command example:
Train with Bhatt Loss
$ python3 train.py -l b -tsid_idx 0
$ python3 train.py -l b -tsid_idx 1
$ python3 train.py -l b -tsid_idx 2

Train with MSE loss
$ python3 train.py -l mse -tsid_idx 0

Resume training
$ python3 train.py -l b -rt True -tsid_idx 0
'''

import sys
sys.path.append('../../../')
from data_converters.utils import *

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
import os
from os import path
import copy

from collections import deque, namedtuple
from itertools import count
from PIL import Image
import shutil
import psutil
import gc
import statistics
import cv2
import csv
from scipy import signal
import json
from itertools import permutations
from queue import Queue
import pickle
import datetime
from collections import defaultdict
from csv import reader
import glob
from PIL import Image
import argparse

# lstm autoencoder reconstruct and predict sequence
from numpy import array
# import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import GRU, LSTM, Bidirectional
from keras.layers import concatenate, Concatenate, Add, Layer
from keras.layers import Dense, MaxPooling1D, Conv1D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Masking
from keras.layers import Lambda
from keras.layers import BatchNormalization
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
'''
Terminology
X: short for X-Encoder, which includes convolution bidirectional layers for reconstruction.

X2: in addition to the losses in X 1) self-reconstruction, 2) cross-reconstruction and 3)
    full-reconstruction., we have 4) fused-reconstruction and 5) multi-reconstruction.

    # ---------------
    #  IMU19 Decoder
    # ---------------
    # de_IMU19 = RepeatVector(C.recent_K)(in_fused_add)
    de_IMU19 = Bidirectional(LSTM(C.IMU19_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(in_fused_add) # de_IMU19)


X8: in addition to X2, both BBX and IMU decoders share the same architecture, which consists of a Conv1D layer and
    one Stacked BiLSTM layers(equivalent to two consecutive BiLSTM layers).

X20: in addition to X8, X20 also considers FTM.
X22: in addition to X20, X22 changes One-to-others loss into Cross-Domain loss that enforces the model to learn to
    reconstruct the data from one domain (such as Cam) to the other (e.g. Phone).
'''
# ----------------------------------------
#  Configurations of the whole experiment
# ----------------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default='0', help='0-14') # edit
        self.parser.add_argument('-k', '--recent_K', type=int, default=10, help='Window length') # edit
        self.parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss')
        self.parser.add_argument('-rt', '--resume_training', action='store_true', help='resume from checkpoint')
        self.args = self.parser.parse_args()

        self.root_path = '../../../..'

        # ------------------------------------------
        #  To be updated in prepare_training_data()
        self.model_id = 'X22_indoor_BBX5in_IMU19_FTM2_' # edit
        if self.args.loss == 'mse':self.model_id += 'test_idx_' + str(self.args.test_seq_id_idx)
        elif self.args.loss == 'b': self.model_id = self.model_id[:self.model_id.index('FTM2_') + len('FTM_2')] + \
            'Bloss_test_idx_' + str(self.args.test_seq_id_idx)
        print('self.model_id: ', self.model_id)
        # self.seq_root_path = self.root_path + '/Data/datasets/RAN/seqs/indoor'
        # self.seq_root_path_for_model = self.root_path + '/RAN4model/seqs/scene0'
        self.seq4model_root_path = self.root_path + '/Data/datasets/RAN4model/seqs/scene0'
        if not os.path.exists(self.seq4model_root_path):
            os.makedirs(self.seq4model_root_path)

        print('self.seq4model_root_path: ', self.seq4model_root_path)
        self.seq_id_path_ls = sorted(glob.glob(self.seq4model_root_path + '/*'))
        self.seq_id_ls = sorted([seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls])
        self.seq_id = self.seq_id_ls[0]
        self.test_seq_id = self.seq_id_ls[self.args.test_seq_id_idx]

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255)
        }

        # --------------------------------------
        #  To be updated in update_parameters()
        self.seq_path = self.seq_id_path_ls[0]
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)

        self.img_type = 'RGBh_ts16_dfv2'
        self.img_path = self.seq_path + '/' + self.img_type
        self.RGBh_ts16_dfv3_ls = []

        self.subjects = [15, 46, 77, 70, 73]
        self.phone_time_offsets = [0] * len(self.subjects)

        print(); print() # debug
        print('self.subjects: ', self.subjects)

        # ----------------------------------------------
        #  Synchronized data: BBXC3,BBX5,IMU,_sync_dfv3
        # ----------------------------------------------
        self.BBXC3_sync_dfv3, self.BBX5_sync_dfv3, self.IMU19_sync_dfv3, self.FTM2_sync_dfv3= [], [], [], []
        self.seq_id = self.seq_id_ls[0]
        self.seq_path_for_model = self.seq4model_root_path + '/' + self.seq_id
        self.sync_dfv3_path = self.seq_path_for_model + '/sync_ts16_dfv3'
        if not os.path.exists(self.sync_dfv3_path): os.makedirs(self.sync_dfv3_path)
        self.BBXC3_sync_dfv3_path = self.sync_dfv3_path + '/BBXC3H_sync_dfv3.pkl'
        self.BBX5_sync_dfv3_path = self.sync_dfv3_path + '/BBX5H_sync_dfv3.pkl'
        self.IMU19_sync_dfv3_path = self.sync_dfv3_path + '/IMU19_sync_dfv3.pkl'
        self.FTM2_sync_dfv3_path = self.sync_dfv3_path + '/FTM2_sync_dfv3.pkl'

        # ------
        #  BBX5
        # ------
        self.BBX5_dim = 5
        self.BBX5_dummy = [0] * self.BBX5_dim

        # ------------
        #  FTM2
        # ------------
        self.FTM2_dim = 2
        self.FTM2_dummy = [0] * self.FTM2_dim

        # -----
        #  IMU
        # -----
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv3_path = self.seq_path + '/IMU_dfv3' # ts13_dfv3 with offsets (os)
        if not os.path.exists(self.IMU_dfv3_path): os.makedirs(self.IMU_dfv3_path)

        # -------
        #  IMU19
        # -------
        self.IMU19_data_types = ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMU19_dim = 3 + 3 + 3 + 4 + 3 + 3 # 19
        self.IMU19_dummy = [0] * self.IMU19_dim

        # --------------
        #  Video Window
        # --------------
        self.crr_ts16_dfv3_ls_all_i = 0
        self.video_len = 0 # len(self.ts12_BBX5_all)
        self.recent_K = self.args.recent_K
        self.n_wins = 0

        self.sub_tracklets = None # (win_i, subj_i, first_f_i_in_win_in_view, last_f_i_in_win_in_view) with len <= K

        # -------
        #  Model
        # -------
        self.n_batch = 32 # 32
        self.n_epochs = 1000000000000000 # 200 # 100000 # 100000
        self.h_BBX5_dim = 32 # X8: 32
        self.h_FTM2_dim = 32
        self.h_IMU19_dim = 32 # X8: 32
        self.h_fused_dim = 32 # X8: 32
        self.n_filters = 32 # X8: 32
        self.kernel_size = 16 # X8: 16
        self.seq_in_BBX5, self.seq_in_IMU19, self.seq_in_FTM2 = None, None, None
        self.seq_out_BBX5, self.seq_out_IMU19, self.seq_out_FTM2 = None, None, None
        self.model = None
        self.checkpoint_root_path = self.root_path + '/Data/checkpoints/' + self.model_id # exp_id
        if not os.path.exists(self.checkpoint_root_path): os.makedirs(self.checkpoint_root_path)
        self.model_path_to_save = self.checkpoint_root_path + '/model.h5'
        self.model_weights_path_to_save = self.checkpoint_root_path + '/w.ckpt'
        self.start_training_time = ''
        self.start_training_time_ckpt_path = ''
        self.history_callback_path_to_save = self.checkpoint_root_path + '/history_callback.p' # self.seq_path + '/' + self.model_id + '_history_callback.p'
        self.history_callback = None
        self.loss_lambda = 1
        # self.opt = None
        # self.learning_rate = 0.1 # edit
        self.save_weights_interval = 2
        self.model_checkpoint = ModelCheckpoint(self.model_weights_path_to_save, \
            monitor='loss', verbose=1, \
            save_weights_only=True, \
            save_best_only=True, mode='auto', \
            period=self.save_weights_interval)
        #  To de updated in prepare_training_data()
        # ------------------------------------------

        # ---------------
        #  Visualization
        # ---------------
        self.vis = False # edit

        # -------
        #  Color
        # -------
        self.color_ls = ['crimson', 'lime green', 'royal blue', 'chocolate', 'purple', 'lemon']
        self.color_dict = {
            'crimson': (60,20,220),
            'lime green': (50,205,50),
            'royal blue': (225,105,65),
            'chocolate': (30,105,210),
            'purple': (128,0,128),
            'lemon': (0,247,255)
        }

C = Config()


class ZeroPadding(Layer):
     def __init__(self, **kwargs):
          super(ZeroPadding, self).__init__(**kwargs)
     def call(self, x, mask=None):
          return K.zeros_like(x)
     def get_output_shape_for(self, input_shape):
          return input_shape

def MyNet():
    # ---------------
    #  BBX5 Encoder
    # ---------------
    in_BBX5 = Input(shape=(C.recent_K, C.BBX5_dim,))
    conv1_BBX5 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_BBX5)
    en_BBX5 = Bidirectional(LSTM(C.h_BBX5_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_BBX5) # in_BBX5)

    # --------------------
    #  FTM2 Encoder
    # --------------------
    in_FTM2 = Input(shape=(C.recent_K, C.FTM2_dim,))
    conv1_FTM2 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_FTM2)
    en_FTM2 = Bidirectional(LSTM(C.h_FTM2_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_FTM2)


    # ---------------
    #  IMU19 Encoder
    # ---------------
    in_IMU19 = Input(shape=(C.recent_K, C.IMU19_dim,))
    conv1_IMU19 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_IMU19)
    en_IMU19 = Bidirectional(LSTM(C.h_IMU19_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(conv1_IMU19) # in_IMU19)

    # ----------------------
    #  Joint Representation
    # ----------------------
    in_fused_add = Add()([en_BBX5, en_IMU19])
    # in_fused_concat = Concatenate()([en_BBX5, en_IMU19])
    # in_fused = RepeatVector(C.recent_K)(in_fused_concat)

    # --------------
    #  BBX5 Decoder
    # --------------
    # de_BBX5 = RepeatVector(C.recent_K)(in_fused_add)
    de_BBX5 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_fused_add)
    de_BBX5 = Bidirectional(LSTM(C.h_fused_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_BBX5) # in_fused_add) # in_fused)
    de_BBX5 = Bidirectional(LSTM(C.BBX5_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_BBX5) # in_fused_add) # de_BBX5)
    de_BBX5 = TimeDistributed(Dense(C.BBX5_dim))(de_BBX5)

    # --------------------
    #  FTM2 Decoder
    # --------------------
    # de_FTM2 = RepeatVector(C.recent_K)(in_fused_add)
    de_FTM2 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_fused_add)
    de_FTM2 = Bidirectional(LSTM(C.h_fused_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_FTM2)
    de_FTM2 = Bidirectional(LSTM(C.FTM2_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_FTM2)
    de_FTM2 = TimeDistributed(Dense(C.FTM2_dim))(de_FTM2)

    # ---------------
    #  IMU19 Decoder
    # ---------------
    # de_IMU19 = RepeatVector(C.recent_K)(in_fused_add)
    de_IMU19 = Conv1D(filters=C.n_filters, kernel_size=C.kernel_size, strides=1,
                       activation='relu', padding='same')(in_fused_add)
    de_IMU19 = Bidirectional(LSTM(C.h_fused_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_IMU19) # in_fused_add) # in_fused)
    de_IMU19 = Bidirectional(LSTM(C.IMU19_dim, activation='relu', \
            batch_input_shape=(C.n_batch, ), return_sequences=True))(de_IMU19) # de_IMU19)
    de_IMU19 = TimeDistributed(Dense(C.IMU19_dim))(de_IMU19)

    # ------------
    #  Base Model
    # ------------
    BaseNet = Model([in_BBX5, in_FTM2, in_IMU19], [de_BBX5, de_FTM2, de_IMU19])
    print('BaseNet.summary(): ', BaseNet.summary())

    # Env0: Encoder for BBX5, Dec0: Decoder for BBX5
    # Env1: Encoder for FTM2, Dec1: Decoder for FTM2
    # Env2: Encoder for IMU19, Dec2: Decoder for IMU19
    # -------------------------------------------
    #  Self-Reconstruction
    #      BBX5 = Dec0(Env0(BBX5))
    #      FTM2 = Dec1(Env1(FTM2))
    #      IMU19 = Dec2(Env2(IMU19))
    # -------------------------------------------
    [sl_rec_BBX5_0, _, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [_, sl_rec_FTM2_1, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [_, _, sl_rec_IMU19_2] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])

    # -------------------------------------------
    #  Cross-Reconstruction
    #      FTM2 = Dec1(Env0(BBX5))
    #      IMU19 = Dec2(Env0(BBX5))
    #      BBX5 = Dec0(Env1(FTM2))
    #      IMU19 = Dec2(Env1(FTM2))
    #      BBX5 = Dec0(Env2(IMU19))
    #      FTM2 = Dec1(Env2(IMU19))
    # -------------------------------------------
    [_, cr_rec_FTM2_3, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [_, _, cr_rec_IMU19_4] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [cr_rec_BBX5_5, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [_, _, cr_rec_IMU19_6] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [cr_rec_BBX5_7, _, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])
    [_, cr_rec_FTM2_8, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])

    # ------------------------------------------------------------------
    #  Fused-Reconstruction
    #      BBX5 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    #      FTM2 = Dec1(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    #      IMU19 = Dec2(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    # ------------------------------------------------------------------
    [fs_rec_BBX5_9, _, _] = BaseNet([in_BBX5, in_FTM2, in_IMU19])
    [_, fs_rec_FTM2_10, _] = BaseNet([in_BBX5, in_FTM2, in_IMU19])
    [_, _, fs_rec_IMU19_11] = BaseNet([in_BBX5, in_FTM2, in_IMU19])

    # ------------------------------------------------------------------------------------
    #  One-to-All-Reconstruction
    #      BBX5 = Dec0(Env0(BBX5)), FTM2 = Dec1(Env0(BBX5)), IMU19 = Dec2(Env0(BBX5))
    #      BBX5 = Dec0(Env1(FTM2)), FTM2 = Dec1(Env1(FTM2)), IMU19 = Dec2(Env1(FTM2))
    #      BBX5 = Dec0(Env2(IMU19)), FTM2 = Dec1(Env2(IMU19)), IMU19 = Dec2(Env2(IMU19))
    # ------------------------------------------------------------------------------------
    [ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU19_14] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU19_17] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU19_20] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])

    # ---------------------------------------------------------
    #  Cross-Domain-Reconstruction
    #      FTM2 = Dec1(Env0(BBX5)), IMU19 = Dec2(Env0(BBX5))
    #      BBX5 = Dec0(Env1(FTM2)), BBX5 = Dec0(Env2(IMU19))
    # ---------------------------------------------------------
    [_, crd_rec_FTM2_21, crd_rec_IMU19_22] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [crd_rec_BBX5_23, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, in_IMU19])

    # -------------------------------------------
    #  Multi-Reconstruction
    #      BBX5 = Dec0(Env0(BBX5))
    #      FTM2 = Dec1(Env0(BBX5))
    #      IMU19 = Dec2(Env0(BBX5))
    #      BBX5 = Dec0(Env1(FTM2))
    #      FTM2 = Dec1(Env1(FTM2))
    #      IMU19 = Dec2(Env1(FTM2))
    #      BBX5 = Dec0(Env2(IMU19))
    #      FTM2 = Dec1(Env2(IMU19))
    #      IMU19 = Dec2(Env2(IMU19))
    # -------------------------------------------
    [mu_rec_BBX5_24, _, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [_, mu_rec_FTM2_25, _] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [_, _, mu_rec_IMU19_26] = BaseNet([in_BBX5, ZeroPadding()(in_FTM2), ZeroPadding()(in_IMU19)])
    [mu_rec_BBX5_27, _, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [_, mu_rec_FTM2_28, _] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [_, _, mu_rec_IMU19_29] = BaseNet([ZeroPadding()(in_BBX5), in_FTM2, ZeroPadding()(in_IMU19)])
    [mu_rec_BBX5_30, _, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])
    [_, mu_rec_FTM2_31, _] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])
    [_, _, mu_rec_IMU19_32] = BaseNet([ZeroPadding()(in_BBX5), ZeroPadding()(in_FTM2), in_IMU19])

    # ------------------------------------------------------------------
    #  Full-Reconstruction
    #      BBX5 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    #      FTM2 = Dec0(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    #      IMU19 = Dec1(Env0(BBX5), Env1(FTM2), Env2(IMU19))
    # ------------------------------------------------------------------
    [fl_rec_BBX5_27, fl_rec_FTM2_27, fl_rec_IMU19_27] = BaseNet([in_BBX5, in_FTM2, in_IMU19])

    def Bhatt_loss(y_true, y_pred):
        small_num = 0.000001
        # print('np.shape(y_true): ', np.shape(y_true)) # debug # e.g. (None, 10, 2)
        y_true, y_pred = tf.cast(y_true, dtype='float64'), tf.cast(y_pred, dtype='float64')
        mu_true, sig_true = tf.cast(y_true[:,:,0], dtype='float64'), tf.cast(y_true[:,:,1], dtype='float64')
        mu_pred, sig_pred = tf.cast(y_pred[:,:,0], dtype='float64'), tf.cast(y_pred[:,:,1], dtype='float64')
        # print('np.shape(mu_true): ', np.shape(mu_true), ', np.shape(sig_true): ', np.shape(sig_true))
        # print('mu_true: ', mu_true, ', sig_true: ', sig_true)
        term0 = tf.math.truediv(tf.math.log(tf.math.add(tf.math.add(tf.math.truediv(\
                        tf.math.truediv(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num), 4.), \
                        tf.math.truediv(tf.math.pow(sig_pred, 2), tf.math.pow(sig_true, 2) + small_num)), 2.) + small_num), 4.)
        term1 = tf.math.truediv(tf.math.truediv(tf.math.pow((mu_true - mu_pred), 2), tf.math.add(tf.math.pow(sig_true, 2), tf.math.pow(sig_pred, 2) + small_num)), 4.)
        return tf.reduce_mean((term0 + term1)) #, axis=-1)  # Note the `axis=-1`

    if C.args.loss == 'mse':
        C.model = Model([in_BBX5, in_FTM2, in_IMU19], \
            [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU19_2, \
            cr_rec_FTM2_3, cr_rec_IMU19_4, cr_rec_BBX5_5, \
            cr_rec_IMU19_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
            fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU19_11, \
            ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU19_14, \
            ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU19_17, \
            ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU19_20, \
            crd_rec_FTM2_21, crd_rec_IMU19_22, crd_rec_BBX5_23, \
            mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU19_26, \
            mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU19_29, \
            mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU19_32])

        # C.opt = keras.optimizers.Adam(learning_rate=C.learning_rate)
        C.model.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse', ],
            loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1, \
                            1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
                            1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1,
                            1, 1, 1,  1, 1, 1,  1, 1, 1,   1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer='adam') # C.opt) # 'adam')
    elif C.args.loss == 'b':
        C.model = Model([in_BBX5, in_FTM2, in_IMU19], \
            [sl_rec_BBX5_0, sl_rec_FTM2_1, sl_rec_IMU19_2, \
            cr_rec_FTM2_3, cr_rec_IMU19_4, cr_rec_BBX5_5, \
            cr_rec_IMU19_6, cr_rec_BBX5_7, cr_rec_FTM2_8, \
            fs_rec_BBX5_9, fs_rec_FTM2_10, fs_rec_IMU19_11, \
            ota_rec_BBX5_12, ota_rec_FTM2_13, ota_rec_IMU19_14, \
            ota_rec_BBX5_15, ota_rec_FTM2_16, ota_rec_IMU19_17, \
            ota_rec_BBX5_18, ota_rec_FTM2_19, ota_rec_IMU19_20, \
            crd_rec_FTM2_21, crd_rec_IMU19_22, crd_rec_BBX5_23, \
            mu_rec_BBX5_24, mu_rec_FTM2_25, mu_rec_IMU19_26, \
            mu_rec_BBX5_27, mu_rec_FTM2_28, mu_rec_IMU19_29, \
            mu_rec_BBX5_30, mu_rec_FTM2_31, mu_rec_IMU19_32, \
            sl_rec_FTM2_1, cr_rec_FTM2_3, cr_rec_FTM2_8, \
            fs_rec_FTM2_10, ota_rec_FTM2_13, ota_rec_FTM2_16, \
            ota_rec_FTM2_19, crd_rec_FTM2_21, mu_rec_FTM2_25, \
            mu_rec_FTM2_28, mu_rec_FTM2_31, fl_rec_FTM2_27])

        # C.opt = keras.optimizers.Adam(learning_rate=C.learning_rate)
        C.model.compile(loss=['mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                'mse', 'mse', 'mse',  'mse', 'mse', 'mse', \
                                Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss, \
                                Bhatt_loss, Bhatt_loss, Bhatt_loss,  Bhatt_loss, Bhatt_loss, Bhatt_loss],
            loss_weights=[1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
                          1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1, \
                          1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1], optimizer='adam') # C.opt) # 'adam')
    # plot_model(BaseNet, show_shapes=True, to_file=str(C.model_id + '_base_net.png'))
    # plot_model(C.model, show_shapes=True, to_file=str(C.model_id + '.png'))
    return C.model

def prepare_training_data():
    seq_in_BBX5_dfv3_ls, seq_in_FTM2_dfv3_ls, seq_in_IMU19_dfv3_ls = [], [], []
    '''
    C.BBX5_sync_dfv3 e.g. (589, 5, 5)
    C.IMU19_sync_dfv3 e.g. (589, 5, 19)
    '''
    C.img_type = 'RGBh_ts16_dfv2'
    print(); print() # debug
    print('C.seq_id_ls: ', C.seq_id_ls)
    # ---------------------------------------
    #  Iterate Over All Train Seq_id - Start
    # ---------------------------------------
    for C.seq_id_idx, C.seq_id in enumerate(C.seq_id_ls):
        if C.seq_id != C.test_seq_id:
            C.seq_path = C.seq_id_path_ls[C.seq_id_idx]
            C.img_path = C.seq_path + '/' + C.img_type

            C.seq_date = C.seq_id[:8]
            C.seq_path_for_model = C.seq4model_root_path + '/' + C.seq_id
            # C.img_path = C.seq_path + '/' + C.img_type
            C.RGBh_ts16_dfv3_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv3_ls.json'
            with open(C.RGBh_ts16_dfv3_ls_path, 'r') as f:
                C.RGBh_ts16_dfv3_ls = json.load(f)
                print(C.RGBh_ts16_dfv3_ls_path, 'loaded!')
                print('C.RGBh_ts16_dfv3_ls[:5]: ', C.RGBh_ts16_dfv3_ls[:5])

            # start_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv3(C.start_ots), C.RGB_ts16_dfv3_ls)).index(True)
            # end_ts16_i = list(map(lambda i: i > ots26_to_ts16_dfv3(C.end_ots), C.RGB_ts16_dfv3_ls)).index(True)
            # print(); print() # debug
            # print('ots26_to_ts16_dfv3(C.start_ots): ', ots26_to_ts16_dfv3(C.start_ots))
            # print('C.RGB_ts16_dfv3_ls[start_ts16_i]: ', C.RGB_ts16_dfv3_ls[start_ts16_i])
            # print('ots26_to_ts16_dfv3(C.end_ots): ', ots26_to_ts16_dfv3(C.end_ots))
            # print('C.RGB_ts16_dfv3_ls[end_ts16_i]: ', C.RGB_ts16_dfv3_ls[end_ts16_i])
            # '''
            # e.g.
            # ots26_to_ts16_dfv3(C.start_ots):  1633372099.829258
            # C.RGB_ts16_dfv3_ls[start_ts16_i]:  1633372099.929300
            # ots26_to_ts16_dfv3(C.end_ots):  1633372277.437527
            # C.RGB_ts16_dfv3_ls[end_ts16_i]:  1633372277.537532
            # '''
            # C.RGB_ts16_dfv3_valid_ls = C.RGB_ts16_dfv3_ls[start_ts16_i : end_ts16_i + 1]

            # print(); print() # debug
            # print('C.seq_id: ', C.seq_id, ', C.seq_id: ', C.seq_id)
            # print('C.seq_path: ', C.seq_path)
            # print('C.seq_path_for_model: ', C.seq_path_for_model)
            # print('len(C.RGB_ts16_dfv3_valid_ls): ', len(C.RGB_ts16_dfv3_valid_ls)) # e.g. 1800
            if C.vis: C.img_path = C.seq_path + '/' + C.img_type

            # ------------------------------------------
            #  Synchronized data: BBX5,IMU19_sync_dfv3
            # ------------------------------------------
            C.sync_dfv3_path = C.seq_path_for_model + '/sync_ts16_dfv3'
            # -----------------
            #  Load BBX5 Data
            # -----------------
            C.BBX5_dim = 5
            C.BBX5_sync_dfv3_path = C.sync_dfv3_path + '/BBX5H_sync_dfv3.pkl'
            C.BBX5_sync_dfv3 = pickle.load(open(C.BBX5_sync_dfv3_path, 'rb'))
            C.BBX5_sync_dfv3 = np.nan_to_num(C.BBX5_sync_dfv3, nan=0)
            print(); print() # debug
            print('np.shape(C.BBX5_sync_dfv3): ', np.shape(C.BBX5_sync_dfv3))
            # e.g. (535, 5, 5)

            # -----------------
            #  Load IMU19 Data
            # -----------------
            C.IMU19_dim = 3 + 3 + 3 + 4 + 3 + 3 # 19
            C.IMU19_data_types = ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
            C.IMU19_sync_dfv3_path = C.sync_dfv3_path + '/IMU19_sync_dfv3.pkl'
            C.IMU19_sync_dfv3 = pickle.load(open(C.IMU19_sync_dfv3_path, 'rb'))
            print(); print() # debug
            print('np.shape(C.IMU19_sync_dfv3): ', np.shape(C.IMU19_sync_dfv3))
            # e.g. (535, 5, 19)

            # ----------------
            #  Load FTM2 Data
            # ----------------
            C.FTM2_dim = 2
            C.FTM2_sync_dfv3_path = C.sync_dfv3_path + '/FTM_sync_dfv3.pkl'
            C.FTM2_sync_dfv3 = pickle.load(open(C.FTM2_sync_dfv3_path, 'rb'))
            print(); print() # debug
            print('np.shape(C.FTM2_sync_dfv3): ', np.shape(C.FTM2_sync_dfv3))
            # e.g. (535, 5, 2)

            # --------------
            #  Video Window
            # --------------
            C.crr_ts16_dfv3_ls_all_i = 0
            C.video_len = len(C.RGBh_ts16_dfv3_ls) # len(C.ts12_BBX5_all)
            print(); print() # debug
            print('C.video_len: ', C.video_len) # e.g. 1800
            C.n_wins = C.video_len - C.recent_K + 1
            print('C.n_wins: ', C.n_wins) # e.g. 1791

            # --------------
            #  Prepare BBX5
            # --------------
            curr_in_view_i_ls = []
            for win_i in range(C.n_wins):
                for subj_i in range(len(C.subjects) - 1):
                    # seq_in_BBX5_ = C.BBX5_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                    seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                    # print(); print() # debug
                    # print('np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
                    # print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
                    # print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
                    # print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
                    # print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
                    # print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
                    '''
                    e.g.
                    seq_in_BBX5_[:, 0]:  [641. 631. 618. 604. 592. 583. 577. 570. 565. 562.]
                    seq_in_BBX5_[:, 1]:  [635. 630. 627. 623. 619. 615. 611. 607. 604. 602.]
                    seq_in_BBX5_[:, 2]:  [1.73513258 1.75361669 1.78351653 1.84246898 1.86370301 1.86906254
                     1.90441883 1.93990803 1.98963535 2.04343772]
                    seq_in_BBX5_[:, 3]:  [157. 152. 147. 146. 147. 148. 149. 145. 142. 140.]
                    seq_in_BBX5_[:, 4]:  [163. 173. 180. 188. 195. 203. 211. 218. 225. 228.]
                    '''
                    # -----------------------------------------------------------------------------
                    #  Note that RGB_ts16_dfv3_valid_ls only works for New Dataset in this version
                    # -----------------------------------------------------------------------------
                    # if C.vis:
                    #     subj_i_RGB_ots26_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
                    #     print(); print() # debug
                    #     print('subj_i_RGB_ots26_img_path: ', subj_i_RGB_ots26_img_path)
                    #     img = cv2.imread(subj_i_RGB_ots26_img_path)
                    #     print(); print() # debug
                    #     print(C.subjects[subj_i], ', np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
                    #     print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
                    #     print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
                    #     print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
                    #     print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
                    #     print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
                    #     '''
                    #     e.g.
                    #     Sid , np.shape(seq_in_BBX5_):  (10, 5)
                    #     seq_in_BBX5_[:, 0]:  [866. 833. 809.   0.   0.   0.   0. 676. 653. 638.]
                    #     seq_in_BBX5_[:, 1]:  [427. 427. 432.   0.   0.   0.   0. 446. 451. 485.]
                    #     seq_in_BBX5_[:, 2]:  [9.25371265 9.26818466 8.887537   0.         0.         0.
                    #      0.         7.5010891  8.03569031 8.17784595]
                    #     seq_in_BBX5_[:, 3]:  [40. 35. 32.  0.  0.  0.  0. 34. 64. 67.]
                    #     seq_in_BBX5_[:, 4]:  [ 46.  46.  62.   0.   0.   0.   0.  63.  77. 144.]
                    #     '''
                    #     subj_color = C.color_dict[C.color_ls[subj_i]]
                    #
                    #     for k_i in range(C.recent_K):
                    #         top_left = (int(seq_in_BBX5_[k_i, 0]) - int(seq_in_BBX5_[k_i, 3] / 2), \
                    #                     int(seq_in_BBX5_[k_i, 1]) - int(seq_in_BBX5_[k_i, 4] / 2))
                    #         bottom_right = (int(seq_in_BBX5_[k_i, 0]) + int(seq_in_BBX5_[k_i, 3] / 2), \
                    #                     int(seq_in_BBX5_[k_i, 1]) + int(seq_in_BBX5_[k_i, 4] / 2))
                    #         img = cv2.circle(img, (int(seq_in_BBX5_[k_i, 0]), int(seq_in_BBX5_[k_i, 1])), 4, subj_color, 4) # Note (col, row) or (x, y) here
                    #         img = cv2.rectangle(img, top_left, bottom_right, subj_color, 2)
                    #     cv2.imshow('img', img); cv2.waitKey(0)

                    curr_in_view_i_ls.append(win_i * 5 + subj_i)
                    seq_in_BBX5_dfv3_ls.append(seq_in_BBX5_)

            # ---------------
            #  Prepare IMU19
            # ---------------
            for win_i in range(C.n_wins):
                for subj_i in range(len(C.subjects) - 1):
                    curr_in_view_i = win_i * 5 + subj_i
                    if curr_in_view_i in curr_in_view_i_ls:
                        # seq_in_BBX5_ = C.BBX5_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                        seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        # print('seq_in_BBX5_: ', seq_in_BBX5_)
                        '''
                        e.g.
                        [[ 434.         2384.            9.43317604   51.           69.        ]
                         [ 450.         2398.            8.5320549    45.           76.        ]
                         [ 448.         2418.            8.58677673   40.           73.        ]
                         [ 447.         2443.            8.39226532   45.           70.        ]
                         [ 451.         2463.            8.08877563   44.           69.        ]
                         [ 456.         2486.            7.67414522   45.           76.        ]
                         [ 459.         2515.            7.3971715    42.           70.        ]
                         [ 473.         2545.            7.31567383   50.           94.        ]
                         [ 503.         2568.            7.58291388   58.          146.        ]
                         [ 507.         2594.            7.5857439    88.          153.        ]]
                        [[ 434.         2384.            9.43317604   51.           69.        ]
                         [ 450.         2398.            8.5320549    45.           76.        ]
                         [ 448.         2418.            8.58677673   40.           73.        ]
                         [ 447.         2443.            8.39226532   45.           70.        ]
                         [ 451.         2463.            8.08877563   44.           69.        ]
                         [ 456.         2486.            7.67414522   45.           76.        ]
                         [ 459.         2515.            7.3971715    42.           70.        ]
                         [ 473.         2545.            7.31567383   50.           94.        ]
                         [ 503.         2568.            7.58291388   58.          146.        ]
                         [ 507.         2594.            7.5857439    88.          153.        ]]
                        '''
                        k_start_i = C.recent_K
                        for k_i in range(C.recent_K):
                            # print(type(seq_in_BBX5_[k_i])) # numpy.ndarray
                            if 0 not in seq_in_BBX5_[k_i]:
                                k_start_i = k_i
                                break

                        # seq_in_IMU19_ = C.IMU19_sync_dfv3[subj_i, win_i : win_i + C.recent_K, :] # old
                        seq_in_IMU19_ = C.IMU19_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        for k_i in range(C.recent_K):
                            if k_i < k_start_i:
                                IMU19_not_in_view = np.full((1, C.IMU19_dim), 0)
                                # print('IMU19_not_in_view: ', IMU19_not_in_view)
                                seq_in_IMU19_[k_i] = IMU19_not_in_view

                        # --------------------------------------------
                        # When subject appears in the middle of the K
                        #   frames, not from the beginning.
                        # --------------------------------------------
                        # if k_start_i > 0:
                        #     print('seq_in_IMU19_: ', seq_in_IMU19_)
                            '''
                            e.g.
                            [[ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00]
                             [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00]
                             [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00]
                             [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
                               0.0000000e+00  0.0000000e+00  0.0000000e+00]
                             [-2.0954866e+00  3.2508876e+00  8.5232540e+00 -7.2049400e-01
                               2.8158462e+00  9.3695260e+00  1.1338525e+00 -1.1182821e+00
                               2.2577858e-01  6.7508490e-01  6.4768410e-02  1.3363415e-01
                               7.2263926e-01  2.2208359e+01 -7.8468760e+00 -3.8774540e+01
                              -9.3414760e-02  8.6078510e-03 -1.0858424e+00]
                             [-8.0696640e-01  2.3522854e+00  1.0723923e+01 -1.1488200e+00
                               2.3024473e+00  9.4665220e+00 -3.2529104e-01 -2.2907019e-01
                              -1.1917534e+00  7.4159560e-01  4.8845995e-02  1.1562438e-01
                               6.5900004e-01  2.0745518e+01 -3.9974870e+00 -3.9055250e+01
                               6.3710630e-02 -4.4039783e-01 -7.2205400e-01]
                             [ 5.6552687e-03  1.8982091e+00  9.4670710e+00 -1.1488200e+00
                               2.3024473e+00  9.4665220e+00 -6.4176387e-01  3.9283370e-01
                               3.7391663e-02  8.5440310e-01  8.4577160e-02  8.6982355e-02
                               5.0524867e-01  1.7119225e+01  3.5399957e+00 -4.1333750e+01
                               1.6544344e-01 -2.1935607e-01 -1.1007553e+00]
                             [-6.5787860e-01  2.1945550e+00  9.7042140e+00 -1.0713353e+00
                               2.4716730e+00  9.4328780e+00  6.6444260e-01 -2.0192194e-01
                              -1.2005806e-01  9.2537254e-01  9.9322550e-02  9.3838334e-02
                               3.5357487e-01  1.2295586e+01  4.5765010e+00 -4.3084816e+01
                              -1.4268380e-01 -1.3759781e-01 -2.6505628e-01]
                             [-2.4698522e+00  7.7978987e-01  1.1293171e+01 -1.0752205e+00
                               1.9969125e+00  9.5442310e+00  1.2478797e+00 -8.7572860e-01
                              -1.0399151e+00  9.3259054e-01  8.3217960e-02  1.0491423e-01
                               3.3517560e-01  1.4684637e+01  5.6806455e+00 -4.1336370e+01
                              -6.7318505e-01  1.9875900e-01 -6.6650970e-02]
                             [-1.4726695e-01  2.0846767e+00  1.0316169e+01 -1.1731023e+00
                               1.8297582e+00  9.5661870e+00 -9.1754220e-01 -5.4648890e-01
                              -6.8620110e-01  9.1922814e-01  4.7161426e-02  7.8359134e-02
                               3.8295600e-01  1.3390116e+01  7.0594115e+00 -4.0558987e+01
                              -3.1737524e-01 -1.8553947e-01 -8.0239765e-02]]
                             '''
                        # print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_)) # e.g. (10, 19)
                        seq_in_IMU19_dfv3_ls.append(seq_in_IMU19_)

                        # >>> FTM2 >>>
                        seq_in_FTM2_ = C.FTM2_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
                        for k_i in range(C.recent_K):
                            if k_i < k_start_i:
                                FTM2_not_in_view = np.full((1, C.FTM2_dim), 0)
                                seq_in_FTM2_[k_i] = FTM2_not_in_view
                        seq_in_FTM2_dfv3_ls.append(seq_in_FTM2_)
                        # <<< FTM2 <<<

            print(); print() # debug
            print('len(seq_in_IMU19_dfv3_ls): ', len(seq_in_IMU19_dfv3_ls))
            print('len(seq_in_FTM2_dfv3_ls): ', len(seq_in_FTM2_dfv3_ls))

    # -------------------------------------
    #  Iterate Over All Train Seq_id - End
    # -------------------------------------
    C.seq_in_BBX5 = np.array(seq_in_BBX5_dfv3_ls)
    C.seq_out_BBX5 = copy.deepcopy(C.seq_in_BBX5)
    C.seq_in_BBX5 = tf.convert_to_tensor(C.seq_in_BBX5)
    C.seq_out_BBX5 = tf.convert_to_tensor(C.seq_out_BBX5)
    print('np.shape(C.seq_in_BBX5): ', np.shape(C.seq_in_BBX5)) # e.g. (27376, 10, 5)
    print('np.shape(C.seq_out_BBX5): ', np.shape(C.seq_out_BBX5))

    C.seq_in_FTM2 = np.array(seq_in_FTM2_dfv3_ls)
    C.seq_out_FTM2 = copy.deepcopy(C.seq_in_FTM2)
    C.seq_in_FTM2 = tf.convert_to_tensor(C.seq_in_FTM2)
    C.seq_out_FTM2 = tf.convert_to_tensor(C.seq_out_FTM2)
    print('np.shape(C.seq_in_FTM2): ', np.shape(C.seq_in_FTM2)) # e.g. (27376, 10, 2)
    print('np.shape(C.seq_out_FTM2): ', np.shape(C.seq_out_FTM2))

    C.seq_in_IMU19 = np.array(seq_in_IMU19_dfv3_ls)
    C.seq_out_IMU19 = copy.deepcopy(C.seq_in_IMU19)
    C.seq_in_IMU19 = tf.convert_to_tensor(C.seq_in_IMU19)
    C.seq_out_IMU19 = tf.convert_to_tensor(C.seq_out_IMU19)
    print('np.shape(C.seq_in_IMU19): ', np.shape(C.seq_in_IMU19)) # e.g. (27376, 10, 19)
    print('np.shape(C.seq_out_IMU19): ', np.shape(C.seq_out_IMU19))

    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_FTM2)[1]
    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_IMU19)[1]
    assert np.shape(C.seq_in_BBX5)[2] == C.BBX5_dim
    assert np.shape(C.seq_in_FTM2)[2] == C.FTM2_dim
    assert np.shape(C.seq_in_IMU19)[2] == C.IMU19_dim

    print(); print() # debug
    print('seq_id_path_ls: ', C.seq_id_path_ls)
    print('seq_id_ls: ', C.seq_id_ls)
    print('C.BBX5_sync_dfv3_path: ', C.BBX5_sync_dfv3_path)

def train():
    # ------------------------------
    #  Load model weights if exists
    # ------------------------------
    if C.args.resume_training:
        # if path.exists(C.model_weights_path_to_save):
        C.model.load_weights(C.model_weights_path_to_save)
        print(C.model_weights_path_to_save, 'loaded!')

    C.start_training_time = str(datetime.datetime.now())
    C.start_training_time_ckpt_path = C.checkpoint_root_path + '/start_training_time_' \
        + str(datetime.datetime.now().strftime('Y%Y_Mth%m_D%d_H%H_Mn%M'))
    pickle.dump(None, open(C.start_training_time_ckpt_path, 'wb'))
    print(C.start_training_time_ckpt_path, 'logged!')

    if C.args.loss == 'mse':
        C.history_callback = C.model.fit([C.seq_in_BBX5, C.seq_in_FTM2, C.seq_in_IMU19], \
            [C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_FTM2, C.seq_out_IMU19, C.seq_out_BBX5, \
            C.seq_out_IMU19, C.seq_out_BBX5, C.seq_out_FTM2, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_FTM2, C.seq_out_IMU19, C.seq_out_BBX5, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19], \
            validation_split=0.1, batch_size=C.n_batch, epochs=C.n_epochs, verbose=2, \
            callbacks=[C.model_checkpoint])
    elif C.args.loss == 'b':
        C.history_callback = C.model.fit([C.seq_in_BBX5, C.seq_in_FTM2, C.seq_in_IMU19], \
            [C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_FTM2, C.seq_out_IMU19, C.seq_out_BBX5, \
            C.seq_out_IMU19, C.seq_out_BBX5, C.seq_out_FTM2, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_FTM2, C.seq_out_IMU19, C.seq_out_BBX5, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_BBX5, C.seq_out_FTM2, C.seq_out_IMU19, \
            C.seq_out_FTM2, C.seq_out_FTM2, C.seq_out_FTM2, \
            C.seq_out_FTM2, C.seq_out_FTM2, C.seq_out_FTM2, \
            C.seq_out_FTM2, C.seq_out_FTM2, C.seq_out_FTM2, \
            C.seq_out_FTM2, C.seq_out_FTM2, C.seq_out_FTM2], \
            validation_split=0.1, batch_size=C.n_batch, epochs=C.n_epochs, verbose=2, \
            callbacks=[C.model_checkpoint])
    pickle.dump(C.history_callback, open(C.history_callback_path_to_save, 'wb'))
    print(C.history_callback_path_to_save, 'saved!')

# -------
#  Start
# -------
C.model = MyNet()
print(C.model.summary())

# --------------
#  Prepare Data
# --------------
prepare_training_data()

# -------
#  Train
# -------
train()

C.model.save_weights(C.model_weights_path_to_save)
print(C.model_weights_path_to_save, 'saved!')

def plot_history():
    print((C.history_callback.history.keys()))
    plt.plot(C.history_callback.history['loss'])
    plt.plot(C.history_callback.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

plot_history()

'''
Ref:
Custom loss
https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
https://towardsdatascience.com/a-comprehensive-guide-to-correlational-neural-network-with-keras-3f7886028e4a
https://machinelearningmastery.com/lstm-autoencoders/
https://theaiacademy.blogspot.com/2020/05/a-comprehensive-guide-to-correlational.html
'''
