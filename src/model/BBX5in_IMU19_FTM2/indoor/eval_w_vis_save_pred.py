from __future__ import division
'''
Command example:
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -wd_ls 1 1 1 0 -tsid_idx 2
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -wd_ls 1 1 1 0 -f_d b -tsid_idx 2

python3 eval_w_vis_save_pred.py -fps 10 -tt random -k 10 -wd_ls 1 1 1 0 -tsid_idx 12
python3 eval_w_vis_save_pred.py -fps 10 -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
python3 eval_w_vis_save_pred.py -fps 10 -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
python3 eval_w_vis_save_pred.py -fps 10 -tsid_idx 1 -tt crowded -k 10 -wd_ls 1 1 1 0
python3 eval_w_vis_save_pred.py -fps 10 -f_d b -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
python3 eval_w_vis_save_pred.py -fps 10 -f_d b -tsid_idx 1 -tt crowded -k 10 -wd_ls 1 1 1 0

Eval the model trained with Bhatt Loss
$ python3 eval_w_vis_save_pred.py -fps 10 -l b -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
$ python3 eval_w_vis_save_pred.py -fps 10 -l b -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
$ python3 eval_w_vis_save_pred.py -fps 10 -l b -tsid_idx 1 -tt crowded -k 10 -wd_ls 1 1 1 0
$ python3 eval_w_vis_save_pred.py -fps 10 -l b -f_d b -tsid_idx 1 -tt random -k 10 -wd_ls 1 1 1 0
$ python3 eval_w_vis_save_pred.py -fps 10 -l b -f_d b -tsid_idx 1 -tt crowded -k 10 -wd_ls 1 1 1 0

Code mofidied from X22_BBX5in_IMU19_FTM2/eval_w_trained_scene1/eval_w_vis_save_pred.py
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
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
X20: in addition to X8, X20 also consider FTM.
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
        self.parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int, default=0) # edit
        self.parser.add_argument('-tt', '--test_type', type=str, default='random', help='random | crowded') # edit
        self.parser.add_argument('-fps', '--fps', type=int, default=10, help='10 | 3 | 1') # edit
        self.parser.add_argument('-v', '--vis', action='store_true', help='Visualization')
        self.parser.add_argument('-vo', '--vis_Others', action='store_true', help='Visualization')
        self.parser.add_argument('-k', '--recent_K', type=int, default=10, help='Window length') # edit
        self.parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss')
        self.parser.add_argument('-bw', '--best_weight', action='store_true')
        self.parser.add_argument('-f_d', '--FTM_dist', type=str, default='eucl', \
            help='eucl: Euclidean Distane;' \
                 'sb: Simplified Bhattacharyya Distance'
                 'b: Bhattacharyya Distance;' \
                 'b_exp1: Bhattacharyya Distance with second term to be exponential.')
        self.parser.add_argument('-dt', '--dist_type', type=str, default='eucl', \
            help='exp: Exponential Distance')
        self.parser.add_argument(
            '-wd_ls',
            "--weight_of_distance_list",  # name on the CLI - drop the `--` for positional/required parameters
            nargs="*",  # 0 or more values expected => creates a list
            type=int,
            default=[1, 1, 1, 0],  # default if nothing is provided
            help='Weight of Distance for Different Modalities in this order -- ' \
                    '0: BBX5, 1: IMU, 2: FTM, 3: D_FTM'
        )
        self.parser.add_argument('-nl', '--noise_level', type=float, default=0.0, help='0.0, 0.1, 0.3, 0.5')
        self.args = self.parser.parse_args()

        self.root_path = '../../../..'

        # ------------------------------------------
        #  To be updated in prepare_training_data()
        self.model_id = 'X22_indoor_BBX5in_IMU19_FTM2_' # edit
        if self.args.loss == 'mse':self.model_id += 'test_idx_' + str(self.args.test_seq_id_idx)
        elif self.args.loss == 'b': self.model_id = self.model_id[:self.model_id.index('FTM2_') + len('FTM_2')] + \
            'Bloss_test_idx_' + str(self.args.test_seq_id_idx)
        print('self.model_id: ', self.model_id)
        self.seq4model_root_path = self.root_path + '/Data/datasets/RAN4model/seqs/scene0'
        if not os.path.exists(self.seq4model_root_path): os.makedirs(self.seq4model_root_path)

        self.seq_root_path_for_model = self.root_path + '/Data/datasets/RAN4model/seqs'
        self.exp_root_path_for_model = self.seq_root_path_for_model + '/exps'
        self.exp_id = 'exp0/indoor'
        self.exp_path_for_model = self.exp_root_path_for_model + '/' + self.exp_id
        if not os.path.exists(self.exp_path_for_model): os.makedirs(self.exp_path_for_model)

        print('self.seq_root_path_for_model: ', self.seq_root_path_for_model)
        self.seq_id_path_ls = sorted(glob.glob(self.seq4model_root_path + '/*'))
        self.seq_id_ls = sorted([seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls])
        self.seq_id = self.seq_id_ls[0]
        self.test_seq_id = self.seq_id_ls[self.args.test_seq_id_idx]
        print('self.test_seq_id: ', self.test_seq_id)

        # >>> Normalize weights to sum up to 1 >>>
        self.w_dct = {'Cam': defaultdict(), 'Phone': defaultdict()}
        self.w_ls = self.args.weight_of_distance_list

        Cam_deno = float(self.w_ls[0] + self.w_ls[3])
        self.w_dct['Cam'] = {'BBX5': float(self.w_ls[0]) / float(Cam_deno), 'D_FTM': float(self.w_ls[3]) / float(Cam_deno)}
        Phone_deno = float(self.w_ls[1] + self.w_ls[2] + self.w_ls[3])
        self.w_dct['Phone'] = {'IMU19': float(self.w_ls[1]) / float(Phone_deno),
                                'FTM2': float(self.w_ls[2]) / float(Phone_deno),
                                'D_FTM': float(self.w_ls[3]) / float(Phone_deno)}
        # <<< Normalize weights to sum up to 1 <<<

        # ---------------
        #  Visualization
        # ---------------
        self.vis = self.args.vis # edit
        self.vis_Others = self.args.vis_Others # edit
        self.RGB_ts16_dfv3_path = ''
        self.vis_Cam_ID_ls = ['A', 'B', 'C', 'D', 'E']

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

        self.img_type = 'RGBh_ts16_dfv4_anonymized'
        self.img_path = self.seq_path + '/' + self.img_type
        self.RGBh_ts16_dfv3_ls = []

        self.subjects = [15, 46, 77, 70, 73]
        self.phone_time_offsets = [0] * len(self.subjects)

        print(); print() # debug
        print('self.subjects: ', self.subjects)

        # self.seq_path = self.seq_root_path + '/scene' + str(self.scene_id) + '/' + self.seq_id
        # self.seq_date = self.seq_id[:8]
        # self.seq_id_to_start_end_ots_offsets = defaultdict()
        # self.subj_to_offset = defaultdict()
        # self.start_ots, self.end_ots, self.subj_to_offset = get_start_end_ts_update_phone_with_offsets(self.seq_id, self.meta_path)
        # self.seq_path_for_model = self.seq_root_path_for_model + '/scene' + str(self.scene_id) + '/' + self.seq_id

        # self.img_path = self.seq_path + '/' + self.img_type
        self.img_path = '../../../../Data/datasets/RAN/seqs/indoor/scene0/' + self.test_seq_id + '/' + self.img_type
        print('self.img_path: ', self.img_path)

        # self.RGB_ts16_dfv3_ls_path = self.seq_path_for_model + '/RGB_ts16_dfv3_ls.json'
        # with open(self.RGB_ts16_dfv3_ls_path, 'r') as f:
        #     self.RGB_ts16_dfv3_ls = json.load(f)
        # self.RGB_ts16_dfv3_valid_ls = self.RGB_ts16_dfv3_ls[self.RGB_ts16_dfv3_ls.index(ots26_to_ts16_dfv3(self.start_ots)) : self.RGB_ts16_dfv3_ls.index(ots26_to_ts16_dfv3(self.end_ots)) + 1]

        # ----------------------------------------------------------
        #  Synchronized data: BBXC3,BBX5,Others_dfv3,IMU,_sync_dfv3
        # ----------------------------------------------------------
        self.BBXC3_sync_dfv3, self.BBX5_sync_dfv3, self.IMU19_sync_dfv3, self.FTM2_sync_dfv3= [], [], [], []
        # self.BBXC3_Others_sync_dfv3, self.BBX5_Others_sync_dfv3 = [], []
        self.seq_id_idx = 0
        self.seq_id = self.seq_id_ls[self.seq_id_idx]
        self.seq_path_for_model = self.seq4model_root_path + '/' + self.seq_id
        self.sync_dfv3_path = self.seq_path_for_model + '/sync_ts16_dfv3'
        if not os.path.exists(self.sync_dfv3_path): os.makedirs(self.sync_dfv3_path)

        # self.Others_id_ls = []
        # self.Others_id_ls_path = self.sync_dfv3_path + '/Others_id_ls.pkl'

        # self.BBXC3_Others_sync_dfv3_path = self.sync_dfv3_path + '/BBXC3_Others_sync_dfv3.pkl'
        # self.BBX5_Others_sync_dfv3_path = self.sync_dfv3_path + '/BBX5_Others_sync_dfv3.pkl'
        if not os.path.exists(self.sync_dfv3_path): os.makedirs(self.sync_dfv3_path)

        self.BBXC3_sync_dfv3_path = self.sync_dfv3_path + '/BBXC3H_sync_dfv3.pkl'
        self.BBX5_sync_dfv3_path = self.sync_dfv3_path + '/BBX5H_sync_dfv3.pkl'
        self.IMU19_sync_dfv3_path = self.sync_dfv3_path + '/IMU19_sync_dfv3.pkl'
        self.FTM2_sync_dfv3_path = self.sync_dfv3_path + '/FTM_sync_dfv3.pkl'
        # self.BBXC3_others_sync_dfv3_path = self.sync_dfv3_path + '/BBXC3_others_sync_dfv3.pkl'
        # self.Other_to_BBX5_sync_dfv3_path = self.sync_dfv3_path + '/Other_to_BBX5_sync_dfv3.pkl'
        if not os.path.exists(self.sync_dfv3_path): os.makedirs(self.sync_dfv3_path)

        # ------
        #  BBX5
        # ------
        self.BBX5_dim = 5
        self.BBX5_dummy = [0] * self.BBX5_dim
        self.max_depth = 500

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
        self.IMU19_range_dct = {'max_arr': None, 'min_arr': None, 'range_arr': None} # for noise
        self.IMU19_range_dct_path = self.exp_path_for_model + '/IMU19_range_dct.json'
        with open(self.IMU19_range_dct_path, 'r') as f:
            self.IMU19_range_dct = json.load(f)
            print(self.IMU19_range_dct_path, 'loaded!')

        # --------------
        #  Video Window
        # --------------
        self.crr_ts16_dfv3_ls_all_i = 0
        self.video_len = 0 # len(self.ts12_BBX5_all)
        self.recent_K = self.args.recent_K
        self.n_wins = 0

        self.seq_subj_i_in_view_dict = defaultdict()
        self.seq_in_BBX5_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BBXC3_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_FTM2_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_IMU19_dict = defaultdict() # key: (win_i, subj_i)
        # self.seq_in_BBX5_Others_dict = defaultdict() # key: (win_i, subj_)
        # self.seq_in_BBXC3_Others_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BBX5_r_shape = None
        self.seq_in_BBXC3_r_shape = None
        self.seq_in_FTM2_c_shape = None
        self.seq_in_IMU19_c_shape = None
        # self.seq_in_BBX5_Others_r_shape = None
        # self.seq_in_BBXC3_Others_r_shape = None

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
        self.checkpoint_root_path = self.root_path + '/Data/checkpoints'
        if not os.path.exists(self.checkpoint_root_path): os.makedirs(self.checkpoint_root_path)
        self.checkpoint_path = self.checkpoint_root_path + '/' + self.model_id # exp_id
        self.model_path_to_save = self.checkpoint_path + '/model.h5'
        if self.args.best_weight: self.model_weights_path_to_save = self.checkpoint_path + '/best_w_dfv3.ckpt'
        else: self.model_weights_path_to_save = self.checkpoint_path + '/w.ckpt'
        self.start_training_time = ''
        self.start_training_time_ckpt_path = ''
        self.history_callback_path_to_save = self.checkpoint_path + '/history_callback.p' # self.seq_path + '/' + self.model_id + '_history_callback.p'
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

        # ----------
        #  Eval Log
        # ----------
        self.log_time = datetime.datetime.now().strftime("%D_%H_%M_%S")
        self.log_time = self.log_time.replace('/', '_')
        print('self.log_time: ', self.log_time)
        self.eval_log_id = ''
        self.eval_log_root_path = self.root_path + '/Data/logs'
        self.eval_log_path = self.eval_log_root_path + '/' + self.model_id
        self.eval_log_file_path = ''
        self.eval_log_file = None

        self.ts16_dfv3_subj_i_to_BBX5_prime = defaultdict()
        self.ts16_dfv3_subj_i_to_BBX5_prime_path_to_save = self.checkpoint_path + '/ts16_dfv3_subj_i_to_BBX5_prime.pkl'
        self.ts16_dfv3_to_pred_BBX5_labels = defaultdict()
        self.ts16_dfv3_to_pred_BBX5_labels_path_to_save = self.checkpoint_path + '/ts16_dfv3_to_pred_BBX5_labels.pkl'

        self.ts16_dfv3_to_eval_stats = defaultdict()
        self.ts16_dfv3_to_eval_stats_path_to_save = self.checkpoint_path + '/ts16_dfv3_to_eval_stats.pkl'

        self.prev_gd_pred_phone_i_BBX_ls, self.prev_gd_pred_phone_i_BBXwDelta_ls, self.prev_gd_pred_phone_i_IMU_ls = [], [], []
        self.prev_hg_pred_phone_i_BBX_ls, self.prev_hg_pred_phone_i_BBXwDelta_ls, self.prev_hg_pred_phone_i_IMU_ls = [], [], []
        self.scene_eval_stats = {'gd': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv3_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv3_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}, \
                                 'hg': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv3_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': np.nan, 'ts16_dfv3_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
        # FN: misses of Phone Holders, TN: Others -> None
        self.scene_eval_stats_path_to_save = self.checkpoint_path + '/scene_eval_stats.pkl'
        #  To de updated in prepare_testing_data()
        # -----------------------------------------


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

# # Scalar Version
# def Bhatt_dist(distr0, distr1, data_type='sample'):
#     small_num = 0.000001
#     # print('distr0: ', distr0, ', distr1: ', distr1)
#     mu0, sig0 = distr0[0], distr0[1]
#     mu1, sig1 = distr1[0], distr1[1]
#     term0 = float(1/4) * math.log(float(1/4) * ( \
#                         float(sig0 ** 2 + small_num) / float(sig1 ** 2 + small_num) + \
#                         float(sig1 ** 2 + small_num) / float(sig0 ** 2 + small_num) + 2))
#     term1 = float(1/4) * ((mu0 - mu1) ** 2 / (sig0 ** 2 + sig1 ** 2 + small_num))
#     return term0 + term1

# Array Version
def Bhatt_dist(mu0, std0, mu1, std1):
    small_num = 0.000001

    var0 = np.add(np.square(std0), small_num)
    var1 = np.add(np.square(std1), small_num)
    # print(); print() # debug
    # print('var0: ', var0)
    # print('np.shape(var0): ', np.shape(var0))
    '''
    e.g.
    [[0.040805 0.082945 0.082945 0.05905  0.05905  0.05905  0.412165 0.412165 0.07129  0.07129 ]]
    (1, 10)
    '''
    sum_diff_var_rates = np.add(np.divide(var0, var1), np.divide(var1, var0))
    log_in = np.multiply(1/4, np.add(sum_diff_var_rates, 2))
    term0 = np.multiply(1/4, np.log(log_in))
    # print(); print() # debug
    # print('np.shape(term0): ', np.shape(term0)) # e.g. (1, 10)

    # sqrt_diff_mu = np.square(np.substract(mu0, mu1))
    sqrt_diff_mu = np.square(mu0 - mu1)
    sum_var = np.add(np.add(np.square(std0), np.square(std1)), small_num)
    term1 = np.multiply(1/4, np.divide(sqrt_diff_mu, sum_var))
    if C.args.dist_type == 'b_exp1': term1 = np.exp(term1)
    return np.add(term0, term1)

def Simplified_Bhatt_dist(mu0, std0, mu1, std1):
    small_num = 0.000001

    var0 = np.add(np.square(std0), small_num)
    var1 = np.add(np.square(std1), small_num)
    # print(); print() # debug
    # print('var0: ', var0)
    # print('np.shape(var0): ', np.shape(var0))
    '''
    e.g.
    [[0.040805 0.082945 0.082945 0.05905  0.05905  0.05905  0.412165 0.412165 0.07129  0.07129 ]]
    (1, 10)
    '''
    term0 = np.add(np.divide(var0, var1), np.divide(var1, var0))
    # print(); print() # debug
    # print('np.shape(term0): ', np.shape(term0)) # e.g. (1, 10)

    # sqrt_diff_mu = np.square(np.substract(mu0, mu1))
    sqrt_diff_mu = np.square(mu0 - mu1)
    sum_var = np.add(np.add(np.square(std0), np.square(std1)), small_num)
    term1 = np.divide(sqrt_diff_mu, sum_var)
    return np.add(term0, term1)

def vis_tracklet(img, seq_in_BBX5_, subj_i):
    # print(); print() # debug
    # print(C.subjects[subj_i], ', np.shape(seq_in_BBX5_): ', np.shape(seq_in_BBX5_)) # e.g. (10, 5)
    # print('seq_in_BBX5_[:, 0]: ', seq_in_BBX5_[:, 0]) # col
    # print('seq_in_BBX5_[:, 1]: ', seq_in_BBX5_[:, 1]) # row
    # print('seq_in_BBX5_[:, 2]: ', seq_in_BBX5_[:, 2]) # depth
    # print('seq_in_BBX5_[:, 3]: ', seq_in_BBX5_[:, 3]) # width
    # print('seq_in_BBX5_[:, 4]: ', seq_in_BBX5_[:, 4]) # height
    '''
    e.g.
    Sid , np.shape(seq_in_BBX5_):  (10, 5)
    seq_in_BBX5_[:, 0]:  [866. 833. 809.   0.   0.   0.   0. 676. 653. 638.]
    seq_in_BBX5_[:, 1]:  [427. 427. 432.   0.   0.   0.   0. 446. 451. 485.]
    seq_in_BBX5_[:, 2]:  [9.25371265 9.26818466 8.887537   0.         0.         0.
     0.         7.5010891  8.03569031 8.17784595]
    seq_in_BBX5_[:, 3]:  [40. 35. 32.  0.  0.  0.  0. 34. 64. 67.]
    seq_in_BBX5_[:, 4]:  [ 46.  46.  62.   0.   0.   0.   0.  63.  77. 144.]
    '''
    subj_color = C.color_dict[C.color_ls[subj_i % len(C.subjects)]]

    # print('C.subjects: ', C.subjects)
    # print('subj: ', subj)

    for k_i in range(C.recent_K):
        if k_i < len(seq_in_BBX5_) - 1:
            top_left = (int(seq_in_BBX5_[k_i, 0]) - int(seq_in_BBX5_[k_i, 3] / 2), \
                        int(seq_in_BBX5_[k_i, 1]) - int(seq_in_BBX5_[k_i, 4] / 2))
            bottom_right = (int(seq_in_BBX5_[k_i, 0]) + int(seq_in_BBX5_[k_i, 3] / 2), \
                        int(seq_in_BBX5_[k_i, 1]) + int(seq_in_BBX5_[k_i, 4] / 2))
            img = cv2.rectangle(img, top_left, bottom_right, subj_color, 2)
            # print('subj_color: ', subj_color, ', subj: ', subj, ', top_left: ', top_left, ', bottom_right: ', bottom_right)
            if k_i == 0: img = cv2.putText(img, C.vis_Cam_ID_ls[subj_i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, subj_color, 2, cv2.LINE_AA)
    return img

def prepare_testing_data():
    # seq_in_BBX5_dfv3_ls, seq_in_BBX5_Others_dfv3_ls = [], []
    seq_in_BBX5_dfv3_ls = []
    seq_in_IMU19_dfv3_ls = []
    seq_in_FTM2_dfv3_ls, seq_in_FTM2_Others_dfv3_ls = [], []
    seq_in_FTM2_dfv3_ls = []
    '''
    C.BBXC3_sync_dfv3 e.g. (589, 5, 3)
    C.IMU19_sync_dfv3 e.g. (589, 5, 19)
    '''
    C.seq_in_BBX5_r_shape = (1, C.recent_K, C.BBX5_dim)
    C.seq_in_FTM2_c_shape = (1, C.recent_K, C.FTM2_dim)
    C.seq_in_IMU19_c_shape = (1, C.recent_K, C.IMU19_dim)
    # C.seq_in_BBX5_Others_r_shape = (1, C.recent_K, C.BBX5_dim)

    print(); print() # debug
    print('C.seq_id_ls: ', C.seq_id_ls)
    # --------------------------------------
    #  Iterate Over One Test Seq_id
    # --------------------------------------
    # for C.seq_id in C.test_seq_id_ls:
    C.seq_idx_idx = C.args.test_seq_id_idx
    C.seq_id = C.seq_id_ls[C.seq_idx_idx] # edit
    print()
    C.seq_path = C.seq_id_path_ls[C.seq_id_idx]

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
    if C.vis:
        C.img_path = '../../../../Data/datasets/RAN/seqs/indoor/scene0/' + C.test_seq_id + '/' + C.img_type
        # print('C.img_path: ', C.img_path)

    # ------------------------------------------
    #  Synchronized data: BBX5,IMU19_sync_dfv3
    # ------------------------------------------
    C.sync_dfv3_path = C.seq_path_for_model + '/sync_ts16_dfv3'
    # ----------------
    #  Load BBX Data
    # ----------------
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
    # e.g. (535, 5, 19)

    # -------------------------------
    #  Load BBX Data for Passers-by
    # -------------------------------
    # C.Others_id_ls_path = C.sync_dfv3_path + '/Others_id_ls.pkl'
    # C.Others_id_ls = pickle.load(open(C.Others_id_ls_path, 'rb'))
    # C.BBX5_Others_sync_dfv3_path = C.sync_dfv3_path + '/BBX5_Others_sync_dfv3.pkl'
    # C.BBX5_Others_sync_dfv3 = pickle.load(open(C.BBX5_Others_sync_dfv3_path, 'rb'))
    # C.BBX5_Others_sync_dfv3 = np.nan_to_num(C.BBX5_Others_sync_dfv3, nan=0)
    # print(); print() # debug
    # print('len(C.Others_id_ls): ',len(C.Others_id_ls))
    # print('np.shape(C.BBX5_Others_sync_dfv3): ', np.shape(C.BBX5_Others_sync_dfv3))
    #
    # C.Others_id_ls_path = C.sync_dfv3_path + '/Others_id_ls.pkl'
    # C.Others_id_ls = pickle.load(open(C.Others_id_ls_path, 'rb'))
    # C.BBX5_Others_sync_dfv3_path = C.sync_dfv3_path + '/BBX5_Others_sync_dfv3.pkl'
    # C.BBX5_Others_sync_dfv3 = pickle.load(open(C.BBX5_Others_sync_dfv3_path, 'rb'))
    # C.BBX5_Others_sync_dfv3 = np.nan_to_num(C.BBX5_Others_sync_dfv3, nan=0)
    # print(); print() # debug
    # print('len(C.Others_id_ls): ',len(C.Others_id_ls))
    # print('np.shape(C.BBX5_Others_sync_dfv3): ', np.shape(C.BBX5_Others_sync_dfv3))

    # --------------
    #  Video Window
    # --------------
    C.crr_ts16_dfv3_ls_all_i = 0
    C.video_len = len(C.RGBh_ts16_dfv3_ls) # len(C.ts16_dfv3_BBX5_all)
    print(); print() # debug
    print('C.video_len: ', C.video_len) # e.g. 1800
    C.n_wins = C.video_len - C.recent_K + 1
    print('C.n_wins: ', C.n_wins) # e.g. 1791

    # -------------------
    #  Prepare Tracklets
    # -------------------
    # -------------
    #  Prepare BBX
    # -------------
    for win_i in range(C.n_wins):
        seq_subj_i_in_view_ls_ = []

        # -----------------------------------------------------------------------------
        #  Note that RGB_ts16_dfv3_valid_ls only works for New Dataset in this version
        # -----------------------------------------------------------------------------
        #  >>> Vis >>>
        if C.vis:
            # subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
            # print('C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]: ', C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1])

            # Last frame of a window
            subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + C.RGBh_ts16_dfv3_ls[win_i + C.recent_K - 1] + '_anonymized.jpg'
            # print(); print() # debug
            # print('subj_i_RGB_ts16_dfv3_img_path: ', subj_i_RGB_ts16_dfv3_img_path)
            img = cv2.imread(subj_i_RGB_ts16_dfv3_img_path)
            # if '20201228' in C.seq_id:
            #     img = img[450:1730, 350:1070]
            # cv2.imshow('img', img); cv2.waitKey(0) # Debug
        #  <<< Vis <<<

        for subj_i in range(len(C.subjects)):
            seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
            # BBX5 in the view
            if seq_in_BBX5_[:, 3].any() != 0 and seq_in_BBX5_[:, 4].any() != 0:
                seq_in_BBX5_dfv3_ls.append(seq_in_BBX5_)
                seq_subj_i_in_view_ls_.append(subj_i)
                C.seq_in_BBX5_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_, axis=0)

            #  >>> Vis >>>
            # if C.vis: vis_tracklet(img, seq_in_BBX5_, C.subjects[subj_i])
            # if C.vis: img = vis_tracklet(img, seq_in_BBX5_, C.subjects[subj_i])
            #  <<< Vis <<<
        #  >>> Vis >>>
        # if C.vis:
        #     cv2.imshow('img', img); cv2.waitKey(0)
        #  <<< Vis <<<
        # C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]] = seq_subj_i_in_view_ls_
        C.seq_subj_i_in_view_dict[C.RGBh_ts16_dfv3_ls[win_i]] = seq_subj_i_in_view_ls_

    # ----------------------
    #  Prepare IMU19 & FTM2
    # ----------------------
    for win_i in range(C.n_wins):
        for subj_i in range(len(C.subjects)):
            print('np.shape(C.IMU19_sync_dfv3): ', np.shape(C.IMU19_sync_dfv3))
            seq_in_IMU19_ = C.IMU19_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
            print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_))
            if len(seq_in_IMU19_) == C.recent_K:
                # >>> Add noise >>>
                if C.args.noise_level > 0.0:
                    noise19 = []
                    for k in range(C.recent_K):
                        noise19_in_k = [np.random.normal(0, C.IMU19_range_dct['range_arr'][d]) * C.args.noise_level for d in range(C.IMU19_dim)]
                        noise19.append(noise19_in_k)
                    noise19 = np.array(noise19)
                    # print(); print() # debug
                    # print('np.shape(noise19): ', np.shape(noise19)) # (10, 19)
                    seq_in_IMU19_ += noise19
                # <<< Add noise <<<

                seq_in_IMU19_dfv3_ls.append(seq_in_IMU19_)
                C.seq_in_IMU19_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU19_, axis=0)

            # >>> FTM2 >>>
            seq_in_FTM2_ = C.FTM2_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
            if len(seq_in_FTM2_) == C.recent_K:
                seq_in_FTM2_dfv3_ls.append(seq_in_FTM2_)
                C.seq_in_FTM2_dict[(win_i, subj_i)] = np.expand_dims(seq_in_FTM2_, axis=0)
            # <<< FTM2 <<<

    # ---------------------
    #  Prepare BBX5 Others
    # ---------------------
    # for win_i in range(C.n_wins):
    #     seq_subj_i_in_view_ls_ = []
    #     #  >>> Vis >>>
    #     if C.vis_Others:
    #         subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
    #         print(); print() # debug
    #         print('subj_i_RGB_ts16_dfv3_img_path: ', subj_i_RGB_ts16_dfv3_img_path)
    #         img = cv2.imread(subj_i_RGB_ts16_dfv3_img_path)
    #     #  <<< Vis <<<
    #     for subj_i_, subj_ in enumerate(C.Others_id_ls):
    #         # print(); print() # debug
    #         # print('subj_i_: ', subj_i_, ', subj_', subj_)
    #         subj_i = subj_i_ + len(C.subjects) - 1
    #         seq_in_BBX5_Others_ = C.BBX5_Others_sync_dfv3[win_i : win_i + C.recent_K, subj_i_, :]
    #         # print('np.shape(seq_in_BBX5_Others_): ', np.shape(seq_in_BBX5_Others_))
    #
    #         # BBX5 in the view
    #         if seq_in_BBX5_Others_[:, 3].any() != 0 and seq_in_BBX5_Others_[:, 4].any() != 0 \
    #             and len(seq_in_BBX5_Others_) == C.recent_K:
    #             # print(); print() # debug
    #             # print('seq_in_BBX5_Others_: ', seq_in_BBX5_Others_)
    #             seq_in_BBX5_Others_dfv3_ls.append(seq_in_BBX5_Others_)
    #             seq_subj_i_in_view_ls_.append(subj_i)
    #             C.seq_in_BBX5_Others_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_Others_, axis=0)
    #
    #         #  >>> Vis >>>
    #         if C.vis_Others: vis_tracklet(img, seq_in_BBX5_Others_, subj_)
    #         #  <<< Vis <<<
    #     #  >>> Vis >>>
    #     if C.vis_Others:
    #         cv2.imshow('img', img); cv2.waitKey(0)
    #     #  <<< Vis <<<
    #     if C.RGB_ts16_dfv3_valid_ls[win_i] not in C.seq_subj_i_in_view_dict:
    #         C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]] = seq_subj_i_in_view_ls_
    #     else:
    #         C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]].extend(seq_subj_i_in_view_ls_)
    #     # print(); print() # debug
    #     # print('C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]]: ', C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]])
    #     # e.g. [0, 1, 2, 10, 11, 13, 14]

    C.seq_in_BBX5 = np.array(seq_in_BBX5_dfv3_ls)
    print(); print() # debug
    print('np.shape(C.seq_in_BBX5): ', np.shape(C.seq_in_BBX5)) # e.g. (5273, 10, 5)
    C.seq_out_BBX5 = copy.deepcopy(C.seq_in_BBX5)

    C.seq_in_IMU19 = np.array(seq_in_IMU19_dfv3_ls)
    print(); print() # debug
    print('np.shape(C.seq_in_IMU19): ', np.shape(C.seq_in_IMU19)) # e.g. (5273, 10, 5)
    C.seq_out_IMU19 = copy.deepcopy(C.seq_in_IMU19)

    # C.seq_in_BBX5_Others = np.array(seq_in_BBX5_Others_dfv3_ls)
    # print(); print() # debug
    # print('np.shape(C.seq_in_BBX5_Others): ', np.shape(C.seq_in_BBX5_Others)) # e.g. (44875, 10, 5)
    # C.seq_out_BBX5_Others = copy.deepcopy(C.seq_in_BBX5_Others)

    C.seq_in_FTM2 = np.array(seq_in_FTM2_dfv3_ls)
    C.seq_out_FTM2 = copy.deepcopy(C.seq_in_FTM2)
    print('np.shape(C.seq_in_FTM2): ', np.shape(C.seq_in_FTM2)) # e.g. (27376, 10, 2)
    print('np.shape(C.seq_out_FTM2): ', np.shape(C.seq_out_FTM2))

    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_IMU19)[1] # == np.shape(C.seq_out_BBX5_Others)[1]
    assert np.shape(C.seq_in_BBX5)[2] == C.BBX5_dim # np.shape(C.seq_out_BBX5_Others)[2] == C.BBX5_dim
    assert np.shape(C.seq_in_IMU19)[2] == C.IMU19_dim

    print(); print() # debug
    print('seq_id_path_ls: ', C.seq_id_path_ls)
    print('seq_id_ls: ', C.seq_id_ls)

    # -------------------------------------------------------------------
    #  Verify if nan_cnt are consistent between BBX5 and IMU19 after the
    #    previous steps.
    # -------------------------------------------------------------------
    BBX5_nan_cnt = 0
    for win_i in range(np.shape(C.seq_in_BBX5)[0]):
        # print(C.seq_in_BBX5[win_i])
        if 0 in C.seq_in_BBX5[win_i]:
            BBX5_nan_cnt += 1
    print('BBX5_nan_cnt: ', BBX5_nan_cnt)

    IMU19_nan_cnt = 0
    for win_i in range(np.shape(C.seq_in_BBX5)[0]):
        # print(C.seq_in_BBX5[win_i])
        if 0 in C.seq_in_BBX5[win_i]:
            IMU19_nan_cnt += 1
    print('IMU19_nan_cnt: ', IMU19_nan_cnt)

    # BBX5_Others_nan_cnt = 0
    # for win_i in range(np.shape(C.seq_in_BBX5_Others)[0]):
    #     # print(C.seq_in_BBX5[win_i])
    #     if 0 in C.seq_in_BBX5_Others[win_i]:
    #         BBX5_Others_nan_cnt += 1
    # print('BBX5_Others_nan_cnt: ', BBX5_Others_nan_cnt)

    assert BBX5_nan_cnt == IMU19_nan_cnt

    print(); print() # debug
    print('len(C.seq_subj_i_in_view_dict.keys()): ', len(C.seq_subj_i_in_view_dict.keys())) # e.g. 55
    print('C.RGBh_ts16_dfv3_ls[:5]: ', C.RGBh_ts16_dfv3_ls[:5])

    # ----------
    #  Eval Log
    # ----------
    C.eval_log_path = C.eval_log_root_path + '/' + C.model_id # exp_id
    os.makedirs((C.eval_log_path), exist_ok=True)
    # C.eval_log_id = 'trained_scene1_' + C.test_type + '_' + C.seq_id + '_' + C.log_time
    C.eval_log_id = C.seq_id + '_f_d_' + C.args.FTM_dist + '_nl_' + str(C.args.noise_level) + '_' + C.log_time
    C.eval_log_id += '_w_ls'
    for w in C.args.weight_of_distance_list: C.eval_log_id += '_' + str(w)

    C.eval_log_file_path = C.eval_log_path + '/' + C.eval_log_id + '_dfv3_eval.log'
    C.eval_log_file = open((C.eval_log_file_path), 'a')
    C.eval_log_file.write(str(C.args) + '\n\n')
    C.eval_log_file.flush()

    C.ts16_dfv3_subj_i_to_BBX5_prime = defaultdict()
    C.ts16_dfv3_subj_i_to_BBX5_prime_path_to_save = C.checkpoint_path + '/' + C.eval_log_id + '/ts16_dfv3_subj_i_to_BBX5_prime.pkl'
    C.ts16_dfv3_to_pred_BBX5_labels = defaultdict()
    C.ts16_dfv3_to_pred_BBX5_labels_path_to_save = C.checkpoint_path + '/' + C.eval_log_id + '/ts16_dfv3_to_pred_BBX5_labels.pkl'

    C.ts16_dfv3_to_eval_stats = defaultdict()
    C.ts16_dfv3_to_eval_stats_path_to_save = C.checkpoint_path + '/' + C.eval_log_id + '/ts16_dfv3_to_eval_stats.pkl'

    C.prev_gd_pred_phone_i_BBX_ls, C.prev_gd_pred_phone_i_IMU_ls = [], []
    C.prev_hg_pred_phone_i_BBX_ls, C.prev_hg_pred_phone_i_IMU_ls = [], []
    C.scene_eval_stats = {'gd': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv3_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv3_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}, \
                             'hg': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv3_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': np.nan, 'ts16_dfv3_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
    # FN: misses of Phone Holders, TN: Others -> None
    C.scene_eval_stats_path_to_save = C.checkpoint_path + '/scene_eval_stats.pkl'

# ----------------------
#  Evaluate Association
# ----------------------
def eval_association():
    for win_i in range(C.n_wins):
        ts16_dfv3 = C.RGBh_ts16_dfv3_ls[win_i] # C.RGB_ts16_dfv3_valid_ls[win_i]
        #  >>> Vis >>>
        if C.vis:
            # subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
            # print('C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]: ', C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1])

            # Last frame of a window
            subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + C.RGBh_ts16_dfv3_ls[win_i + C.recent_K - 1 - 1] + '_anonymized.jpg'
            # print(); print() # debug
            # print('subj_i_RGB_ts16_dfv3_img_path: ', subj_i_RGB_ts16_dfv3_img_path)
            img = cv2.imread(subj_i_RGB_ts16_dfv3_img_path)
            # if '20201228' in C.seq_id:
            #     img = img[450:1730, 350:1070]
            # cv2.imshow('img', img); cv2.waitKey(0) # Debug
        #  <<< Vis <<<

        if ts16_dfv3 in C.seq_subj_i_in_view_dict.keys():
            print()
            seq_subj_i_in_view_ls_ = C.seq_subj_i_in_view_dict[ts16_dfv3]
            print('seq_subj_i_in_view_ls_: ', seq_subj_i_in_view_ls_) # e.g. [1, 2, 3, 4]
            print(C.args)

            rand_seq_subj_i_in_view_ls = copy.deepcopy(seq_subj_i_in_view_ls_)
            random.shuffle(rand_seq_subj_i_in_view_ls)
            print('rand_seq_subj_i_in_view_ls: ', rand_seq_subj_i_in_view_ls)

            n = len(seq_subj_i_in_view_ls_)
            if n > 0:
                A_Cam, A_Phone = [], [] # row: BBX5, col: IMU19

                C.ts16_dfv3_subj_i_to_BBX5_prime[ts16_dfv3] = defaultdict()
                # --------------------------------------------------------
                #  Iterate Over All Sujects Present in the Current Window
                # --------------------------------------------------------
                for r_i, subj_i_r in enumerate(seq_subj_i_in_view_ls_):
                    dist_Cam_row, dist_Phone_row = [], []
                    seq_in_BBX5_r = C.seq_in_BBX5_dict[(win_i, subj_i_r)]

                    # print('\n subj_i_r: ', subj_i_r, ', C.subjects[subj_i_r]: ', C.subjects[subj_i_r]); HERE
                    # print('np.shape(seq_in_BBX5_r): ', np.shape(seq_in_BBX5_r));
                    # e.g. (1, 10, 5)
                    #  >>> Vis >>>
                    if C.vis: img = vis_tracklet(img, np.squeeze(seq_in_BBX5_r, axis=0), rand_seq_subj_i_in_view_ls[subj_i_r])
                    #  <<< Vis <<<

                    # if subj_i_r in range(len(C.subjects)):
                    #     seq_in_BBX5_r = C.seq_in_BBX5_dict[(win_i, subj_i_r)]
                    # else:
                    #     seq_in_BBX5_r = C.seq_in_BBX5_Others_dict[(win_i, subj_i_r)]
                    #     # print(); print() # debug
                    #     # print('win_i, subj_i_r: ', win_i, subj_i_r)
                    #     # print('seq_in_BBX5_r: ', seq_in_BBX5_r)

                    # -------------------------
                    #  Iterate Over All Phones
                    # -------------------------
                    for c_i in range(len(C.subjects)):
                        seq_in_IMU19_c = C.seq_in_IMU19_dict[(win_i, c_i)]
                        seq_in_FTM2_c = C.seq_in_FTM2_dict[(win_i, c_i)] / 1000
                        # yhat_from_BBX5_r = C.model.predict([seq_in_BBX5_r, \
                        #     np.zeros((C.seq_in_IMU19_c_shape[0], C.seq_in_IMU19_c_shape[1], \
                        #     C.seq_in_IMU19_c_shape[2]))], batch_size=1)
                        yhat_from_BBX5_r = C.model.predict([seq_in_BBX5_r, \
                            np.zeros((C.seq_in_FTM2_c_shape[0], C.seq_in_FTM2_c_shape[1], \
                            C.seq_in_FTM2_c_shape[2])), \
                            np.zeros((C.seq_in_IMU19_c_shape[0], C.seq_in_IMU19_c_shape[1], \
                            C.seq_in_IMU19_c_shape[2]))], batch_size=1)

                        yhat_from_IMU19_FTM2_c = C.model.predict([np.zeros((C.seq_in_BBX5_r_shape[0], \
                                        C.seq_in_BBX5_r_shape[1], C.seq_in_BBX5_r_shape[2])), \
                                        seq_in_FTM2_c, \
                                        seq_in_IMU19_c], batch_size=1)

                        pred_FTM2_r, pred_IMU19_r = yhat_from_BBX5_r[21] / 1000, yhat_from_BBX5_r[22]
                        pred_BBX5_c = yhat_from_IMU19_FTM2_c[23]
                        # print(); print() # debug
                        # print('np.shape(pred_FTM2_r): ', np.shape(pred_FTM2_r))
                        # print('np.shape(pred_BBX5_c): ', np.shape(pred_BBX5_c))
                        # print('pred_FTM2_r: ', pred_FTM2_r)
                        # print('pred_BBX5_c: ', pred_BBX5_c)
                        # print('seq_in_BBX5_r[:,:,2]: ', seq_in_BBX5_r[:,:,2]) # Depth

                        dist_BBX5 = np.linalg.norm(seq_in_BBX5_r - pred_BBX5_c)
                        if C.args.FTM_dist == 'eucl': dist_FTM2 = np.linalg.norm(seq_in_FTM2_c - pred_FTM2_r)
                        elif C.args.FTM_dist == 'b':
                            dist_FTM2 = np.sum(Bhatt_dist(seq_in_FTM2_c[:,:,0], seq_in_FTM2_c[:,:,1], \
                                                            pred_FTM2_r[:,:,0], pred_FTM2_r[:,:,1]))
                        elif C.args.FTM_dist == 'sb':
                            dist_FTM2 = np.sum(Simplified_Bhatt_dist(seq_in_FTM2_c[:,:,0], seq_in_FTM2_c[:,:,1], \
                                                            pred_FTM2_r[:,:,0], pred_FTM2_r[:,:,1]))
                        dist_IMU19 = np.linalg.norm(seq_in_IMU19_c - pred_IMU19_r)
                        dist_D_FTM = np.linalg.norm(seq_in_BBX5_r[:,:,2] - seq_in_FTM2_c[:,:,0] / 1000)
                        # print('dist_BBX5: ', dist_BBX5, ', dist_FTM2: ', dist_FTM2, ', dist_IMU19: ', dist_IMU19, ', dist_D_FTM: ', dist_D_FTM)
                        dist_Cam = C.w_dct['Cam']['BBX5'] * dist_BBX5 + C.w_dct['Cam']['D_FTM'] * dist_D_FTM
                        dist_Phone = C.w_dct['Phone']['IMU19'] * dist_IMU19 + C.w_dct['Phone']['FTM2'] * dist_FTM2 + \
                                        C.w_dct['Phone']['D_FTM'] * dist_D_FTM

                        # print('pred_BBX5_c: ', pred_BBX5_c)
                        '''
                        e.g.
                        [[[ 554.79376   2694.2146       7.5205255   88.706406   150.91786  ]
                          [ 557.4017    2660.969        7.2532496   89.636604   156.27997  ]
                          [ 541.8497    2619.2585       7.103594    85.66533    150.42226  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]
                          [ 535.3716    2602.0254       7.021078    83.95114    147.89742  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]
                          [ 533.0839    2591.1797       7.011936    83.62846    147.34201  ]]]
                        '''
                        C.ts16_dfv3_subj_i_to_BBX5_prime[ts16_dfv3][c_i] = pred_BBX5_c

                        # print(); print() # debug
                        # print('r_i: ', r_i, ', c_i: ', c_i, ', dist_BBX5: ', dist_BBX5, ', dist_IMU19: ', dist_IMU19)
                        dist_Cam_row.append(dist_Cam)
                        dist_Phone_row.append(dist_Phone)
                    A_Cam.append(dist_Cam_row)
                    A_Phone.append(dist_Phone_row)

                # ----------------------------------
                #  Compute Eval Stats in One Window
                # ----------------------------------
                # passers_by_num_BBX, passers_by_num_IMU = 0, 0
                # e.g. [0, 1, 31, 34, 36, 37] where 0, 1 are phone holders and 31, 34, 36, 37 are passers-by
                first_other_idx = len(C.subjects)
                # for i, seq_subj_i in enumerate(seq_subj_i_in_view_ls_):
                #     if seq_subj_i >= len(C.subjects) - 1:
                #         first_other_idx = i
                #         # passers_by_num_BBX = passers_by_num_IMU = len(seq_subj_i_in_view_ls_) - i
                #         break

                print(); print() # debug
                print('first_other_idx: ', first_other_idx)
                print('seq_subj_i_in_view_ls_: ', seq_subj_i_in_view_ls_)

                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3] = defaultdict()

                # -----------------
                #  Greedy Matching
                # -----------------
                gd_ts16_dfv3_Cam_correct_num, gd_ts16_dfv3_Phone_correct_num, gd_ts16_dfv3_total_num = \
                    0, 0, len(seq_subj_i_in_view_ls_)
                gd_ts16_dfv3_Cam_TP, gd_ts16_dfv3_Cam_FP, gd_ts16_dfv3_Cam_TN, gd_ts16_dfv3_Cam_FN = 0, 0, 0, 0
                gd_ts16_dfv3_Phone_TP, gd_ts16_dfv3_Phone_FP, gd_ts16_dfv3_Phone_TN, gd_ts16_dfv3_Phone_FN = 0, 0, 0, 0
                C.scene_eval_stats['gd']['Cam']['total_num'] += gd_ts16_dfv3_total_num
                C.scene_eval_stats['gd']['Phone']['total_num'] += gd_ts16_dfv3_total_num
                A_Cam_arr, A_Phone_arr = np.array(A_Cam), np.array(A_Phone)

                # -----
                #  Cam
                # -----
                gd_pred_phone_i_Cam_ls = []
                for i in range(np.shape(A_Cam_arr)[1]):
                    col = list(A_Cam_arr[:, i])
                    # print(); print() # debug
                    # print('col: ', col)
                    gd_pred_phone_i_Cam_ls.append(col.index(min(col)))

                for i, seq_subj_i in enumerate(seq_subj_i_in_view_ls_):
                    if i < first_other_idx: # GT: Phone holders
                        if gd_pred_phone_i_Cam_ls[i] == seq_subj_i:
                            gd_ts16_dfv3_Cam_TP += 1
                            gd_ts16_dfv3_Cam_correct_num += 1
                        elif seq_subj_i in gd_pred_phone_i_Cam_ls: # Predicted as another phone holder
                            gd_ts16_dfv3_Cam_FP += 1
                        elif seq_subj_i not in gd_pred_phone_i_Cam_ls: # Predicted as passer-by
                            gd_ts16_dfv3_Cam_FN += 1
                    else: # GT: Passers-by
                        if seq_subj_i in gd_pred_phone_i_Cam_ls: # Predicted as phone holder
                            gd_ts16_dfv3_Cam_FP += 1
                        else: # Predicted as passer-by
                            gd_ts16_dfv3_Cam_TN += 1
                            gd_ts16_dfv3_Cam_correct_num += 1

                # >>> IDSWITCH >>>
                # if win_i > 0:
                #     for id_i, id in enumerate(gd_pred_phone_i_Cam_ls):
                #         if C.prev_gd_pred_phone_i_Cam_ls[id_i] != id: C.scene_eval_stats['gd']['Cam']['IDSWITCH'] += 1
                # C.prev_gd_pred_phone_i_Cam_ls = gd_pred_phone_i_Cam_ls
                # <<< IDSWITCH <<<
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['gd'] = defaultdict()
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['gd']['gd_pred_phone_i_Cam_ls'] = gd_pred_phone_i_Cam_ls

                # -----
                #  Phone
                # -----
                gd_pred_phone_i_Phone_ls = []
                for i in range(np.shape(A_Phone_arr)[1]):
                    col = list(A_Phone_arr[:, i])
                    # print(); print() # debug
                    # print('col: ', col)
                    gd_pred_phone_i_Phone_ls.append(col.index(min(col)))

                for i, seq_subj_i in enumerate(seq_subj_i_in_view_ls_):
                    if i < first_other_idx: # GT: Phone holders
                        if gd_pred_phone_i_Phone_ls[i] == seq_subj_i:
                            gd_ts16_dfv3_Phone_TP += 1
                            gd_ts16_dfv3_Phone_correct_num += 1
                        elif seq_subj_i in gd_pred_phone_i_Phone_ls: # Predicted as another phone holder
                            gd_ts16_dfv3_Phone_FP += 1
                        elif seq_subj_i not in gd_pred_phone_i_Phone_ls: # Predicted as passer-by
                            gd_ts16_dfv3_Phone_FN += 1
                    else: # GT: Passers-by
                        if seq_subj_i in gd_pred_phone_i_Phone_ls: # Predicted as phone holder
                            gd_ts16_dfv3_Phone_FP += 1
                        else: # Predicted as passer-by
                            gd_ts16_dfv3_Phone_TN += 1
                            gd_ts16_dfv3_Phone_correct_num += 1

                # >>> IDSWITCH >>>
                # if win_i > 0:
                #     for id_i, id in enumerate(gd_pred_phone_i_Phone_ls):
                #         if C.prev_gd_pred_phone_i_Phone_ls[id_i] != id: C.scene_eval_stats['gd']['Phone']['IDSWITCH'] += 1
                # C.prev_gd_pred_phone_i_Phone_ls = gd_pred_phone_i_Phone_ls
                # <<< IDSWITCH <<<
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['gd']['gd_pred_phone_i_Phone_ls'] = gd_pred_phone_i_Phone_ls

                C.scene_eval_stats['gd']['Cam']['TP'] += gd_ts16_dfv3_Cam_TP
                C.scene_eval_stats['gd']['Cam']['FP'] += gd_ts16_dfv3_Cam_FP
                C.scene_eval_stats['gd']['Cam']['TN'] += gd_ts16_dfv3_Cam_TN
                C.scene_eval_stats['gd']['Cam']['FN'] += gd_ts16_dfv3_Cam_FN
                C.scene_eval_stats['gd']['Cam']['correct_num'] += gd_ts16_dfv3_Cam_correct_num
                C.scene_eval_stats['gd']['Cam']['ts16_dfv3_Cam_IDP'] = \
                    format(float(gd_ts16_dfv3_Cam_correct_num) / \
                    float(gd_ts16_dfv3_total_num), '.4f')
                C.scene_eval_stats['gd']['Cam']['cumu_Cam_IDP'] = \
                    format(float(C.scene_eval_stats['gd']['Cam']['correct_num']) / \
                    float(C.scene_eval_stats['gd']['Cam']['total_num']), '.4f')

                C.scene_eval_stats['gd']['Phone']['TP'] += gd_ts16_dfv3_Phone_TP
                C.scene_eval_stats['gd']['Phone']['FP'] += gd_ts16_dfv3_Phone_FP
                C.scene_eval_stats['gd']['Phone']['TN'] += gd_ts16_dfv3_Phone_TN
                C.scene_eval_stats['gd']['Phone']['FN'] += gd_ts16_dfv3_Phone_FN
                C.scene_eval_stats['gd']['Phone']['correct_num'] += gd_ts16_dfv3_Phone_correct_num
                C.scene_eval_stats['gd']['Phone']['ts16_dfv3_Phone_IDP'] = \
                    format(float(gd_ts16_dfv3_Phone_correct_num) / \
                    float(gd_ts16_dfv3_total_num), '.4f')
                C.scene_eval_stats['gd']['Phone']['cumu_Phone_IDP'] = \
                    format(float(C.scene_eval_stats['gd']['Phone']['correct_num']) / \
                    float(C.scene_eval_stats['gd']['Phone']['total_num']), '.4f')


                # -------------------
                #  Linear Assignment
                # -------------------
                row_ind_Cam, col_ind_Cam = linear_sum_assignment(A_Cam)
                row_ind_Phone, col_ind_Phone = linear_sum_assignment(A_Phone)
                # Each value in row_ind_Cam is the index in seq_subj_i_in_view_ls_

                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg'] = defaultdict()
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg']['GND'] = seq_subj_i_in_view_ls_
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg']['row_ind_Cam'] = row_ind_Cam
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg']['col_ind_Cam'] = col_ind_Cam
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg']['row_ind_Phone'] = row_ind_Phone
                C.ts16_dfv3_to_pred_BBX5_labels[ts16_dfv3]['hg']['col_ind_Phone'] = col_ind_Phone

                hg_ts16_dfv3_Cam_correct_num, hg_ts16_dfv3_Phone_correct_num, \
                    hg_ts16_dfv3_total_num = 0, 0, len(seq_subj_i_in_view_ls_)
                hg_ts16_dfv3_Cam_TP, hg_ts16_dfv3_Cam_FP, hg_ts16_dfv3_Cam_TN, hg_ts16_dfv3_Cam_FN = 0, 0, 0, 0
                hg_ts16_dfv3_Phone_TP, hg_ts16_dfv3_Phone_FP, hg_ts16_dfv3_Phone_TN, hg_ts16_dfv3_Phone_FN = 0, 0, 0, 0
                C.scene_eval_stats['hg']['Cam']['total_num'] += hg_ts16_dfv3_total_num
                C.scene_eval_stats['hg']['Phone']['total_num'] += hg_ts16_dfv3_total_num

                # >>> Eval Cam >>>
                # ------------------------
                #  Predicted Cam Identity
                # ------------------------
                hg_ts16_dfv3_Cam_correct_num += np.count_nonzero(col_ind_Cam==seq_subj_i_in_view_ls_)
                # for i, col_i in enumerate(col_ind_Cam):
                #     '''
                #     col_i: Phone col_i predicted as seq_subj_i_in_view_ls_[row_ind_Cam[i]]
                #     while GT is : col_i
                #     '''
                #     # seq_subj_i_in_view_ls_[row_ind_Cam[i]]: prediction for phone holder [col_i]
                #     pred_id = seq_subj_i_in_view_ls_[row_ind_Cam[i]]
                #     if pred_id in range(len(C.subjects)): # Predicted as Phone holders # Positive
                #         if pred_id == col_i: # Correct
                #             # col_i: true identity of phone holders
                #             hg_ts16_dfv3_Cam_TP += 1
                #             hg_ts16_dfv3_Cam_correct_num += 1
                #         else:
                #             hg_ts16_dfv3_Cam_FP += 1
                #     else: # Predicted as Passers-by # Negative
                #         hg_ts16_dfv3_Cam_FN += 1 # GT should be one of the phone holders
                #     # >>> IDSWITCH >>>
                #     # if win_i == 0:
                #     #     if i == 0: C.prev_hg_pred_phone_i_Cam_ls = []
                #     #     C.prev_hg_pred_phone_i_Cam_ls.append(pred_id)
                #     # else:
                #     #     if pred_id != C.prev_hg_pred_phone_i_Cam_ls[i]:
                #     #         C.scene_eval_stats['hg']['Cam']['IDSWITCH'] += 1
                #     #     C.prev_hg_pred_phone_i_Cam_ls[i] = pred_id
                #     # <<< IDSWITCH <<<

                # ----------------------------
                #  Non-Predicted Cam Identity
                # ----------------------------
                # if first_other_idx != len(C.subjects):
                #     for i, seq_subj_i in enumerate(seq_subj_i_in_view_ls_):
                #         if i in row_ind_Cam: continue # skip predictions
                #         if i < first_other_idx: # Predict Phone Holders as Passsers-by
                #             hg_ts16_dfv3_Cam_FN += 1
                #         else: # Predict Passers-by as Passers-by
                #             hg_ts16_dfv3_Cam_TN += 1
                #             hg_ts16_dfv3_Cam_correct_num += 1
                # <<< Eval Cam <<<

                # >>> Eval Phone >>>
                # ------------------------
                #  Predicted Phone Identity
                # ------------------------
                hg_ts16_dfv3_Phone_correct_num += np.count_nonzero(col_ind_Phone==seq_subj_i_in_view_ls_)
                # for i, col_i in enumerate(col_ind_Phone):
                #     '''
                #     col_i: Phone col_i predicted as seq_subj_i_in_view_ls_[row_ind_Phone[i]]
                #     while GT is : col_i
                #     '''
                #     # seq_subj_i_in_view_ls_[row_ind_Phone[i]]: prediction for phone holder [col_i]
                #     pred_id = seq_subj_i_in_view_ls_[row_ind_Phone[i]]
                #     if pred_id in range(len(C.subjects)): # Predicted as Phone holders # Positive
                #         if pred_id == col_i: # Correct
                #             # col_i: true identity of phone holders
                #             hg_ts16_dfv3_Phone_TP += 1
                #             hg_ts16_dfv3_Phone_correct_num += 1
                #         else:
                #             hg_ts16_dfv3_Phone_FP += 1
                #     else: # Predicted as Passers-by # Negative
                #         hg_ts16_dfv3_Phone_FN += 1 # GT should be one of the phone holders
                #     # >>> IDSWITCH >>>
                #     # if win_i == 0:
                #     #     if i == 0: C.prev_hg_pred_phone_i_Phone_ls = []
                #     #     C.prev_hg_pred_phone_i_Phone_ls.append(pred_id)
                #     # else:
                #     #     if pred_id != C.prev_hg_pred_phone_i_Phone_ls[i]:
                #     #         C.scene_eval_stats['hg']['Phone']['IDSWITCH'] += 1
                #     #     C.prev_hg_pred_phone_i_Phone_ls[i] = pred_id
                #     # <<< IDSWITCH <<<

                # ----------------------------
                #  Non-Predicted Phone Identity
                # ----------------------------
                # if first_other_idx != len(C.subjects):
                #     for i, seq_subj_i in enumerate(seq_subj_i_in_view_ls_):
                #         if i in row_ind_Phone: continue # skip predictions
                #         if i < first_other_idx: # Predict Phone Holders as Passsers-by
                #             hg_ts16_dfv3_Phone_FN += 1
                #         else: # Predict Passers-by as Passers-by
                #             hg_ts16_dfv3_Phone_TN += 1
                #             hg_ts16_dfv3_Phone_correct_num += 1
                # <<< Eval Phone <<<

                C.scene_eval_stats['hg']['Cam']['TP'] += hg_ts16_dfv3_Cam_TP
                C.scene_eval_stats['hg']['Cam']['FP'] += hg_ts16_dfv3_Cam_FP
                C.scene_eval_stats['hg']['Cam']['TN'] += hg_ts16_dfv3_Cam_TN
                C.scene_eval_stats['hg']['Cam']['FN'] += hg_ts16_dfv3_Cam_FN
                C.scene_eval_stats['hg']['Cam']['correct_num'] += hg_ts16_dfv3_Cam_correct_num
                C.scene_eval_stats['hg']['Cam']['ts16_dfv3_Cam_IDP'] = \
                    format(float(hg_ts16_dfv3_Cam_correct_num) / \
                    float(hg_ts16_dfv3_total_num), '.4f')
                C.scene_eval_stats['hg']['Cam']['cumu_Cam_IDP'] = \
                    format(float(C.scene_eval_stats['hg']['Cam']['correct_num']) / \
                    float(C.scene_eval_stats['hg']['Cam']['total_num']), '.4f')

                C.scene_eval_stats['hg']['Phone']['TP'] += hg_ts16_dfv3_Phone_TP
                C.scene_eval_stats['hg']['Phone']['FP'] += hg_ts16_dfv3_Phone_FP
                C.scene_eval_stats['hg']['Phone']['TN'] += hg_ts16_dfv3_Phone_TN
                C.scene_eval_stats['hg']['Phone']['FN'] += hg_ts16_dfv3_Phone_FN
                C.scene_eval_stats['hg']['Phone']['correct_num'] += hg_ts16_dfv3_Phone_correct_num
                C.scene_eval_stats['hg']['Phone']['ts16_dfv3_Phone_IDP'] = \
                    format(float(hg_ts16_dfv3_Phone_correct_num) / \
                    float(hg_ts16_dfv3_total_num), '.4f')
                C.scene_eval_stats['hg']['Phone']['cumu_Phone_IDP'] = \
                    format(float(C.scene_eval_stats['hg']['Phone']['correct_num']) / \
                    float(C.scene_eval_stats['hg']['Phone']['total_num']), '.4f')

                # Old
                # la_res_dict = {'ts16_dfv3': ts16_dfv3, 'win_i' : win_i, 'C.n_wins' : C.n_wins, \
                #     'shape(A)' : str(np.shape(np.array(A_BBX5))), \
                #     'seq_subj_i_in_view_ls_' : seq_subj_i_in_view_ls_, \
                #     'gd_pred_phone_i_BBX_ls' : gd_pred_phone_i_BBX_ls, \
                #     'gd_ts16_dfv3_BBX_IDP': C.scene_eval_stats['gd']['BBX']['ts16_dfv3_BBX_IDP'], \
                #     'gd_cumu_BBX_IDP': C.scene_eval_stats['gd']['BBX']['cumu_BBX_IDP'], \
                #     'gd_BBX_IDSWITCH': C.scene_eval_stats['gd']['BBX']['IDSWITCH'], \
                #     'gd_ts16_dfv3_IMU_IDP': C.scene_eval_stats['gd']['IMU']['ts16_dfv3_IMU_IDP'], \
                #     'gd_cumu_IMU_IDP': C.scene_eval_stats['gd']['IMU']['cumu_IMU_IDP'], \
                #     'gd_IMU_IDSWITCH': C.scene_eval_stats['gd']['IMU']['IDSWITCH'], \
                #     'row_ind_BBX5' : row_ind_BBX5, 'col_ind_BBX5' : col_ind_BBX5, \
                #     'hg_ts16_dfv3_BBX_IDP' : C.scene_eval_stats['hg']['BBX']['ts16_dfv3_BBX_IDP'], \
                #     'hg_cumu_BBX_IDP' : C.scene_eval_stats['hg']['BBX']['cumu_BBX_IDP'], \
                #     'hg_BBX_IDSWITCH': C.scene_eval_stats['hg']['BBX']['IDSWITCH'], \
                #     'row_ind_IMU19' : row_ind_IMU19, 'col_ind_IMU19' : col_ind_IMU19, \
                #     'hg_ts16_dfv3_IMU_IDP' : C.scene_eval_stats['hg']['IMU']['ts16_dfv3_IMU_IDP'], \
                #     'hg_cumu_IMU_IDP' : C.scene_eval_stats['hg']['IMU']['cumu_IMU_IDP'], \
                #     'hg_IMU_IDSWITCH': C.scene_eval_stats['hg']['IMU']['IDSWITCH']}

                la_res_dict = {'ts16_dfv3': ts16_dfv3, 'win_i' : win_i, 'C.n_wins' : C.n_wins, \
                    'shape(A)' : str(np.shape(np.array(A_Cam))), \
                    'seq_subj_i_in_view_ls_' : seq_subj_i_in_view_ls_, \
                    'gd_pred_phone_i_Cam_ls' : gd_pred_phone_i_Cam_ls, \
                    'gd_pred_phone_i_Phone_ls' : gd_pred_phone_i_Phone_ls, \
                    'scene_eval_stats': C.scene_eval_stats, \
                    'row_ind_Cam' : row_ind_Cam, 'col_ind_Cam' : col_ind_Cam, \
                    'row_ind_Phone' : row_ind_Phone, 'col_ind_Phone' : col_ind_Phone}

                print(la_res_dict)

                C.ts16_dfv3_to_eval_stats[ts16_dfv3] = la_res_dict
                C.eval_log_file.write(str(la_res_dict) + '\n\n')
                C.eval_log_file.flush()
                # e.g. shape(A):  (3, 5) , seq_subj_i_in_view_ls_:  [0, 1, 4] , row_ind:  [0 1 2] , col_ind:  [3 2 4]

        if C.vis: cv2.imshow('img', img); cv2.waitKey(0)

    print()

    # ---------------
    #  Log Eval Data
    # ---------------
    gd_cumu_Cam_IDP = format(float(C.scene_eval_stats['gd']['Cam']['cumu_Cam_IDP']), '.4f')
    gd_cumu_Phone_IDP = format(float(C.scene_eval_stats['gd']['Phone']['cumu_Phone_IDP']), '.4f')
    hg_cumu_Cam_IDP = format(float(C.scene_eval_stats['hg']['Cam']['cumu_Cam_IDP']), '.4f')
    hg_cumu_Phone_IDP = format(float(C.scene_eval_stats['hg']['Phone']['cumu_Phone_IDP']), '.4f')
    log_str = 'gd_cumu_Cam_IDP_' + gd_cumu_Cam_IDP + \
        '_gd_cumu_Phone_IDP_' + gd_cumu_Phone_IDP + \
        '_hg_cumu_Cam_IDP_' + hg_cumu_Cam_IDP + \
        '_hg_cumu_Phone_IDP_' + hg_cumu_Phone_IDP
    print(log_str)

    C.eval_log_file.write(log_str + '\n')
    C.eval_log_file.flush()

    C.model_weights_path_to_save = C.checkpoint_path + '/' + C.eval_log_id + '_' + log_str + '_w_dfv3.ckpt'
    C.model.save_weights(C.model_weights_path_to_save)
    print(C.model_weights_path_to_save, 'saved!')

    C.ts16_dfv3_subj_i_to_BBX5_prime_path_to_save = C.checkpoint_path + '/' + \
        C.eval_log_id + '_' + log_str + '_dfv3_ts16_dfv3_subj_i_to_BBX5_prime.pkl'
    pickle.dump(C.ts16_dfv3_subj_i_to_BBX5_prime, open(C.ts16_dfv3_subj_i_to_BBX5_prime_path_to_save, 'wb'))
    print(C.ts16_dfv3_subj_i_to_BBX5_prime_path_to_save, 'saved!')

    C.ts16_dfv3_to_pred_BBX5_labels_path_to_save = C.checkpoint_path + '/' + \
        C.eval_log_id + '_' + log_str + '_dfv3_ts16_dfv3_to_pred_BBX5_labels.pkl'
    pickle.dump(C.ts16_dfv3_to_pred_BBX5_labels, open(C.ts16_dfv3_to_pred_BBX5_labels_path_to_save, 'wb'))
    print(C.ts16_dfv3_to_pred_BBX5_labels_path_to_save, 'saved!')

    C.scene_eval_stats_path_to_save = C.checkpoint_path + '/' + \
        C.eval_log_id + '_' + log_str + '_dfv3_scene_eval_stats.pkl'
    pickle.dump(C.scene_eval_stats_path_to_save, open(C.scene_eval_stats_path_to_save, 'wb'))
    print(C.scene_eval_stats_path_to_save, 'saved!')

# -------
#  Start
# -------
if __name__ == '__main__':
    C.model = MyNet(); print(C.model.summary())
    C.model.load_weights(C.model_weights_path_to_save); print(C.model_weights_path_to_save, ' loaded!')
    prepare_testing_data()
    eval_association()
