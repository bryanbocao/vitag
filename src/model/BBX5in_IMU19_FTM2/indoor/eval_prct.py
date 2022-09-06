from __future__ import division
'''
Command examples:
$ python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0
$ python3 eval_prct.py -fps 10 -k 10 -tsid_idx 1
$ python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0 -nl 0.1
$ python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0 -nl 0.3
$ python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0 -nl 0.5
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
from scipy.spatial import distance, procrustes
import argparse

# lstm autoencoder reconstruct and predict sequence
from numpy import array
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
        self.parser.add_argument('-m', '--machine', type=str, default='snow', help='snow | aw | AI') # edit # self.args.machine
        self.parser.add_argument(
            '-trid_sc_idx_ls',
            "--train_scene_id_idx_ls",  # name on the CLI - drop the `--` for positional/required parameters
            nargs="*",  # 0 or more values expected => creates a list
            type=int,
            default=[1],  # default if nothing is provided # e.g. [1, 2, 3, 4]
            help='One-hot encoder of Scene index'
        )
        self.parser.add_argument('-tt', '--test_type', type=str, default='rand_ss', help='rand_ss | crowded') # edit
        self.parser.add_argument('-fps', '--fps', type=int, default=10, help='10 | 3 | 1') # edit
        self.parser.add_argument('-v', '--vis', action='store_true', help='Visualization') # edit
        self.parser.add_argument('-k', '--recent_K', type=int, default=10, help='Window length') # edit
        self.parser.add_argument('-l', '--loss', type=str, default='mse', help='mse: Mean Squared Error | b: Bhattacharyya Loss')
        self.parser.add_argument('-bw', '--best_weight', type=bool, default=False)
        self.parser.add_argument('-f_d', '--FTM_dist', type=str, default='eucl', \
            help='eucl: Euclidean Distane;' \
                 'sb: Simplified Bhattacharyya Distance'
                 'b: Bhattacharyya Distance;' \
                 'b_exp1: Bhattacharyya Distance with second term to be exponential.') # prct
        self.parser.add_argument('-dt', '--dist_type', type=str, default='prct', \
            help='eucl: Euclidean Distance, prct: Procrustes Analysis, exp: Exponential Distance') # prct
        self.parser.add_argument('-tsid_idx', '--test_seq_id_idx', type=int)
        self.parser.add_argument('-nl', '--noise_level', type=float, default=0.0, help='0.0, 0.1, 0.3, 0.5')

        self.args = self.parser.parse_args()

        self.root_path = '../../../..'

        # ------------------------------------------
        #  To be updated in prepare_training_data()
        self.model_id = 'X22_indoor_BBX5in_IMU19_FTM2_prct' # edit
        self.model_id += '_'
        # if self.args.loss == 'mse':self.model_id += 'test_idx_' + str(self.args.test_seq_id_idx)
        # elif self.args.loss == 'b': self.model_id = self.model_id[:self.model_id.index('FTM2_') + len('FTM_2')] + \
        #     'Bloss_test_idx_' + str(self.args.test_seq_id_idx)
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

        self.img_type = 'RGB'

        # ---------------
        #  Visualization
        # ---------------
        self.vis = False # edit
        self.vis_Others = False # edit
        self.vis_eval = False # edit
        self.RGB_ts16_dfv3_path = ''

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

        # self.seq_path = self.seq_root_path + '/scene' + str(self.scene_id) + '/' + self.seq_id
        # self.seq_date = self.seq_id[:8]
        # self.seq_id_to_start_end_ots_offsets = defaultdict()
        # self.subj_to_offset = defaultdict()
        # self.start_ots, self.end_ots, self.subj_to_offset = get_start_end_ts_update_phone_with_offsets(self.seq_id, self.meta_path)
        # self.seq_path_for_model = self.seq_root_path_for_model + '/scene' + str(self.scene_id) + '/' + self.seq_id
        self.img_path = self.seq_path + '/' + self.img_type
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
        # self.Other_to_BBX5_sync_dfv3_path = self.sync_dfv3_path + '/Other_to_BBX5H_sync_dfv3.pkl'
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

        # -------------
        #  IMU_NED_pos
        # -------------
        # >>> prct >>>
        self.IMU_NED_pos_dim = 2
        self.IMU_NED_pos_dummy = [0] * self.IMU_NED_pos_dim
        self.IMU_dNED_pos_range_dct = {'max_arr': None, 'min_arr': None, 'range_arr': None} # for noise
        self.IMU_dNED_pos_range_dct_path = self.exp_path_for_model + '/IMU_dNED_pos_range_dct.json'
        with open(self.IMU_dNED_pos_range_dct_path, 'r') as f:
            self.IMU_dNED_pos_range_dct = json.load(f)
            print(self.IMU_dNED_pos_range_dct_path, 'loaded!')
        # <<< prct <<<

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
        self.seq_in_IMU_NED_pos_dict = defaultdict() # key: (win_i, subj_i) # prct
        # self.seq_in_BBX5_Others_dict = defaultdict() # key: (win_i, subj_)
        # self.seq_in_BBXC3_Others_dict = defaultdict() # key: (win_i, subj_i)
        self.seq_in_BBX5_r_shape = None
        self.seq_in_BBXC3_r_shape = None
        self.seq_in_FTM2_c_shape = None
        self.seq_in_IMU19_c_shape = None
        self.seq_in_IMU_NED_pos_c_shape = None # prct
        self.seq_in_BBX5_Others_r_shape = None
        self.seq_in_BBXC3_Others_r_shape = None

        # -------
        #  Model
        # -------
        self.checkpoint_root_path = self.root_path + '/Data/checkpoints'
        if not os.path.exists(self.checkpoint_root_path): os.makedirs(self.checkpoint_root_path)
        self.checkpoint_path = self.checkpoint_root_path + '/' + self.model_id # exp_id
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

        self.ts16_dfv3_to_eval_stats = defaultdict()
        self.ts16_dfv3_to_eval_stats_path_to_save = self.checkpoint_path + '/ts16_dfv3_to_eval_stats.pkl'

        self.prev_gd_pred_phone_i_BBX_ls, self.prev_gd_pred_phone_i_IMU_ls = [], []
        self.prev_hg_pred_phone_i_BBX_ls, self.prev_hg_pred_phone_i_IMU_ls = [], []
        self.scene_eval_stats = {'gd': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': 0, 'ts16_dfv3_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': 0, 'ts16_dfv3_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}, \
                                 'hg': {'BBX': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': 0, 'ts16_dfv3_BBX_IDP': 0.0, 'cumu_BBX_IDP': 0.0}, \
                                        'IMU': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                            'IDSWITCH': 0, 'ts16_dfv3_IMU_IDP': 0.0, 'cumu_IMU_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
        # FN: misses of Phone Holders, TN: Others -> None
        self.scene_eval_stats_path_to_save = self.checkpoint_path + '/scene_eval_stats.pkl'
        #  To de updated in prepare_testing_data()
        # -----------------------------------------


C = Config()

def prepare_testing_data():
    # seq_in_BBX5_dfv3_ls, seq_in_BBX5_Others_dfv3_ls = [], []
    seq_in_BBX5_dfv3_ls = []
    # seq_in_IMU19_dfv3_ls = []
    seq_in_IMU_NED_pos_dfv3_ls = []
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

    # -----------------------
    #  Load IMU_NED_pos Data
    # -----------------------
    # >>> prct >>>
    C.IMU_NED_pos_sync_dfv3_path = C.sync_dfv3_path + '/IMU_NED_pos_sync_dfv3.pkl'
    C.IMU_NED_pos_sync_dfv3 = pickle.load(open(C.IMU_NED_pos_sync_dfv3_path, 'rb'))
    print(); print() # debug
    print('np.shape(C.IMU_NED_pos_sync_dfv3): ', np.shape(C.IMU_NED_pos_sync_dfv3))
    # e.g. (535, 5, 2)
    # <<< prct <<<

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
            subj_i_RGB_ts16_dfv3_img_path = C.img_path + '/' + ts16_dfv3_to_ots26(C.RGB_ts16_dfv3_valid_ls[win_i + C.recent_K - 1]) + '.png'
            print(); print() # debug
            print('subj_i_RGB_ts16_dfv3_img_path: ', subj_i_RGB_ts16_dfv3_img_path)
            img = cv2.imread(subj_i_RGB_ts16_dfv3_img_path)
        #  <<< Vis <<<

        for subj_i in range(len(C.subjects)): # Note len(C.subjects) for indoor and len(C.subjects) - 1 for outdoor
            seq_in_BBX5_ = C.BBX5_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
            # BBX5 in the view
            if seq_in_BBX5_[:, 3].any() != 0 and seq_in_BBX5_[:, 4].any() != 0:
                seq_in_BBX5_dfv3_ls.append(seq_in_BBX5_)
                seq_subj_i_in_view_ls_.append(subj_i)
                C.seq_in_BBX5_dict[(win_i, subj_i)] = np.expand_dims(seq_in_BBX5_, axis=0)

            #  >>> Vis >>>
            # if C.vis: vis_tracklet(img, seq_in_BBX5_, C.subjects[subj_i])
            if C.vis: vis_tracklet(img, seq_in_BBX5_, C.subjects[subj_i])
            #  <<< Vis <<<
        #  >>> Vis >>>
        if C.vis:
            cv2.imshow('img', img); cv2.waitKey(0)
        #  <<< Vis <<<
        # C.seq_subj_i_in_view_dict[C.RGB_ts16_dfv3_valid_ls[win_i]] = seq_subj_i_in_view_ls_
        C.seq_subj_i_in_view_dict[C.RGBh_ts16_dfv3_ls[win_i]] = seq_subj_i_in_view_ls_

    # ----------------------------------
    #  Prepare Phone3 from IMU19 & FTM2
    # ----------------------------------
    for win_i in range(C.n_wins):
        for subj_i in range(len(C.subjects)): # Note len(C.subjects) for indoor and len(C.subjects) - 1 for outdoor
            # print('np.shape(C.IMU19_sync_dfv3): ', np.shape(C.IMU19_sync_dfv3)) # (1816, 2, 19)
            # seq_in_IMU19_ = C.IMU19_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :]
            # print('np.shape(seq_in_IMU19_): ', np.shape(seq_in_IMU19_)) # e.g. (10, 19)
            seq_in_IMU_NED_pos_ = C.IMU_NED_pos_sync_dfv3[win_i : win_i + C.recent_K, subj_i, :] # prct
            # print('np.shape(seq_in_IMU_NED_pos_): ', np.shape(seq_in_IMU_NED_pos_)) # e.g. (10, 2) # prct
            # if len(seq_in_IMU19_) == C.recent_K:
            if len(seq_in_IMU_NED_pos_) == C.recent_K: # prct
                # >>> Add noise >>>
                if C.args.noise_level > 0.0:
                    # noise19 = []
                    noise_NED_pos = [] # prct
                    for k in range(C.recent_K):
                        # noise19_in_k = [np.random.normal(0, C.IMU19_range['range_arr'][d]) * C.args.noise_level for d in range(C.IMU19_dim)]
                        noise_NED_pos_in_k = [np.random.normal(0, C.IMU_dNED_pos_range_dct['range_arr'][d]) * C.args.noise_level for d in range(C.IMU_NED_pos_dim)] # prct
                        # noise19.append(noise19_in_k)
                        noise_NED_pos.append(noise_NED_pos_in_k) # prct
                    # noise19 = np.array(noise19)
                    noise_NED_pos = np.array(noise_NED_pos) # prct
                    # print(); print() # debug
                    # print('np.shape(noise19): ', np.shape(noise19)) # (10, 19)
                    # seq_in_IMU19_ += noise19
                    seq_in_IMU_NED_pos_ += noise_NED_pos # prct
                # <<< Add noise <<<

                # seq_in_IMU19_dfv3_ls.append(seq_in_IMU19_)
                # C.seq_in_IMU19_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU19_, axis=0)
                seq_in_IMU_NED_pos_dfv3_ls.append(seq_in_IMU_NED_pos_) # prct
                C.seq_in_IMU_NED_pos_dict[(win_i, subj_i)] = np.expand_dims(seq_in_IMU_NED_pos_, axis=0) # prct

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

    # C.seq_in_IMU19 = np.array(seq_in_IMU19_dfv3_ls)
    # print(); print() # debug
    # print('np.shape(C.seq_in_IMU19): ', np.shape(C.seq_in_IMU19)) # e.g. (5273, 10, 5)
    # C.seq_out_IMU19 = copy.deepcopy(C.seq_in_IMU19)

    # >>> prct >>>
    C.seq_in_IMU_NED_pos = np.array(seq_in_IMU_NED_pos_dfv3_ls)
    print(); print() # debug
    print('np.shape(C.seq_in_IMU_NED_pos): ', np.shape(C.seq_in_IMU_NED_pos)) # e.g. (5273, 10, 5)
    C.seq_out_IMU_NED_pos = copy.deepcopy(C.seq_in_IMU_NED_pos)
    # <<< prct <<<

    # C.seq_in_BBX5_Others = np.array(seq_in_BBX5_Others_dfv3_ls)
    # print(); print() # debug
    # print('np.shape(C.seq_in_BBX5_Others): ', np.shape(C.seq_in_BBX5_Others)) # e.g. (44875, 10, 5)
    # C.seq_out_BBX5_Others = copy.deepcopy(C.seq_in_BBX5_Others)

    C.seq_in_FTM2 = np.array(seq_in_FTM2_dfv3_ls)
    C.seq_out_FTM2 = copy.deepcopy(C.seq_in_FTM2)
    print('np.shape(C.seq_in_FTM2): ', np.shape(C.seq_in_FTM2)) # e.g. (27376, 10, 2)
    print('np.shape(C.seq_out_FTM2): ', np.shape(C.seq_out_FTM2))

    assert np.shape(C.seq_in_BBX5)[1] == np.shape(C.seq_in_IMU_NED_pos)[1] # == np.shape(C.seq_out_BBX5_Others)[1]
    assert np.shape(C.seq_in_BBX5)[2] == C.BBX5_dim # np.shape(C.seq_out_BBX5_Others)[2] == C.BBX5_dim
    assert np.shape(C.seq_in_IMU_NED_pos)[2] == C.IMU_NED_pos_dim

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

    IMU_NED_pos_nan_cnt = 0
    for win_i in range(np.shape(C.seq_in_BBX5)[0]):
        # print(C.seq_in_BBX5[win_i])
        if 0 in C.seq_in_BBX5[win_i]:
            IMU_NED_pos_nan_cnt += 1
    print('IMU_NED_pos_nan_cnt: ', IMU_NED_pos_nan_cnt)

    # BBX5_Others_nan_cnt = 0
    # for win_i in range(np.shape(C.seq_in_BBX5_Others)[0]):
    #     # print(C.seq_in_BBX5[win_i])
    #     if 0 in C.seq_in_BBX5_Others[win_i]:
    #         BBX5_Others_nan_cnt += 1
    # print('BBX5_Others_nan_cnt: ', BBX5_Others_nan_cnt)

    assert BBX5_nan_cnt == IMU_NED_pos_nan_cnt

    print('len(C.seq_subj_i_in_view_dict.keys()): ', len(C.seq_subj_i_in_view_dict.keys())) # e.g. 55

    # ----------
    #  Eval Log
    # ----------
    C.eval_log_path = C.eval_log_root_path + '/' + C.model_id # exp_id
    os.makedirs((C.eval_log_path), exist_ok=True)
    # C.eval_log_id = 'trained_scene1_' + C.test_type + '_' + C.seq_id + '_f_d_' + \
    #     C.args.FTM_dist + '_nl_' + str(C.args.noise_level) + '_' + C.log_time
    C.eval_log_id = C.seq_id + '_f_d_' + C.args.FTM_dist + '_nl_' + str(C.args.noise_level) + '_' + C.log_time
    # C.eval_log_id += '_w_ls'

    C.eval_log_file_path = C.eval_log_path + '/' + C.eval_log_id + '_dfv3_eval.log'
    C.eval_log_file = open((C.eval_log_file_path), 'a')
    C.eval_log_file.write(str(C.args) + '\n\n')
    C.eval_log_file.flush()

    C.ts16_dfv3_to_eval_stats = defaultdict()
    C.ts16_dfv3_to_eval_stats_path_to_save = C.checkpoint_path + '/' + C.eval_log_id + '/ts16_dfv3_to_eval_stats.pkl'

    C.prev_gd_pred_phone_i_BBX_ls, C.prev_gd_pred_phone_i_IMU_ls = [], []
    C.prev_hg_pred_phone_i_BBX_ls, C.prev_hg_pred_phone_i_IMU_ls = [], []
    C.scene_eval_stats = {'gd': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': 0, 'ts16_dfv3_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': 0, 'ts16_dfv3_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}, \
                             'hg': {'Cam': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': 0, 'ts16_dfv3_Cam_IDP': 0.0, 'cumu_Cam_IDP': 0.0}, \
                                    'Phone': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'correct_num': 0, 'total_num': 0, \
                                        'IDSWITCH': 0, 'ts16_dfv3_Phone_IDP': 0.0, 'cumu_Phone_IDP': 0.0}}} # hg: Hungarian, gd: greedy-matching
    # FN: misses of Phone Holders, TN: Others -> None
    C.scene_eval_stats_path_to_save = C.checkpoint_path + '/scene_eval_stats.pkl'

# ----------------------
#  Evaluate Association
# ----------------------
def eval_association():
    for win_i in range(C.n_wins):
        ts16_dfv3 = C.RGBh_ts16_dfv3_ls[win_i] # C.RGB_ts16_dfv3_valid_ls[win_i]
        if ts16_dfv3 in C.seq_subj_i_in_view_dict.keys():
            print()
            seq_subj_i_in_view_ls_ = C.seq_subj_i_in_view_dict[ts16_dfv3]
            print('seq_subj_i_in_view_ls_: ', seq_subj_i_in_view_ls_) # e.g. [1, 2, 3, 4]
            print(C.args)

            n = len(seq_subj_i_in_view_ls_)
            if n > 0:
                A_Cam, A_Phone = [], [] # row: BBX5, col: IMU_NED_pos

                # --------------------------------------------------------
                #  Iterate Over All Sujects Present in the Current Window
                # --------------------------------------------------------
                for r_i, subj_i_r in enumerate(seq_subj_i_in_view_ls_):
                    dist_Cam_row, dist_Phone_row = [], []
                    seq_in_BBX5_r = C.seq_in_BBX5_dict[(win_i, subj_i_r)]
                    # if subj_i_r in range(len(C.subjects) - 1):
                    #     seq_in_BBX5_r = C.seq_in_BBX5_dict[(win_i, subj_i_r)]
                    # else:
                    #     seq_in_BBX5_r = C.seq_in_BBX5_Others_dict[(win_i, subj_i_r)]
                    #     # print(); print() # debug
                    #     # print('win_i, subj_i_r: ', win_i, subj_i_r)
                    #     # print('seq_in_BBX5_r: ', seq_in_BBX5_r)

                    seq_in_BBXC3_r = seq_in_BBX5_r[:,:,:3] # prct

                    # -------------------------
                    #  Iterate Over All Phones
                    # -------------------------
                    for c_i in range(len(C.subjects)):
                        # seq_in_IMU19_c = C.seq_in_IMU19_dict[(win_i, c_i)]
                        # print(); print() # debug
                        # print('C.seq_in_IMU_NED_pos_dict.keys(): ', C.seq_in_IMU_NED_pos_dict.keys())
                        seq_in_IMU_NED_pos_c = C.seq_in_IMU_NED_pos_dict[(win_i, c_i)] # prct
                        seq_in_FTM2_c = C.seq_in_FTM2_dict[(win_i, c_i)] / 1000

                        # >>> prct >>>
                        seq_in_FTM2_c_avg_ = np.expand_dims(seq_in_FTM2_c[:,:,0], axis=2)
                        seq_in_FTM2_c_avg = copy.deepcopy(seq_in_FTM2_c_avg_)
                        # seq_in_Phone3_c = np.concatenate((seq_in_IMU_NED_pos_c, seq_in_FTM2_c_avg), axis=2)
                        seq_in_Phone3_c_ = np.concatenate((seq_in_IMU_NED_pos_c, seq_in_FTM2_c_avg), axis=2)
                        seq_in_Phone3_c = copy.deepcopy(seq_in_Phone3_c_)
                        seq_in_Phone3_c[:,:,0] = seq_in_Phone3_c_[:,:,1]
                        seq_in_Phone3_c[:,:,1] = seq_in_Phone3_c_[:,:,0]
                        # print('np.shape(seq_in_Phone3_c): ', np.shape(seq_in_Phone3_c))
                        # <<< prct <<<

                        # print(); print() # debug
                        # print('seq_in_BBXC3_r: ', seq_in_BBXC3_r)
                        # print('seq_in_Phone3_c: ', seq_in_Phone3_c)
                        # print('np.shape(seq_in_BBXC3_r): ', np.shape(seq_in_BBXC3_r))
                        # print('np.shape(seq_in_Phone3_c): ', np.shape(seq_in_Phone3_c))
                        '''
                        e.g. seq_in_BBXC3_r:  [[[1195.          469.            7.34640169]
                          [1168.          456.            7.86406231]
                          [1148.          449.            8.15638638]
                          [1128.          450.            8.2352066 ]
                          [1099.          449.            9.09870338]
                          [1082.          449.            9.08148861]
                          [1070.          448.            9.35062027]
                          [1055.          439.            9.73178196]
                          [1037.          437.           10.06332111]
                          [1028.          428.            9.8390789 ]]]
                        seq_in_Phone3_c:  [[[   0.18457023  -11.45690252    5.587     ]
                          [ -14.27090162    6.96040062    5.587     ]
                          [  46.49393494  -38.23935451    5.587     ]
                          [ -52.30602474   94.19956197    5.587     ]
                          [  25.2560727   -61.10991044    5.334     ]
                          [  28.90025787   16.01405102    5.334     ]
                          [  -6.65145928  -60.69907418    5.334     ]
                          [ 110.59845226 -114.16891373    7.382     ]
                          [  58.0977431     5.23712236    7.382     ]
                          [   4.92400712    4.36235761    7.382     ]]]
                        '''

                        # >>> Phone 2D coordinates Normalization >>>
                        phone_len = np.linalg.norm(seq_in_Phone3_c[0,0,:2] - seq_in_Phone3_c[0,9,:2])
                        if phone_len > 0: # Only needed when the tracklet moves in the phone domain
                            cam_len = np.linalg.norm(seq_in_BBXC3_r[0,0,:2] - seq_in_BBXC3_r[0,9,:2])
                            scale_in_phone = float(cam_len) / float(phone_len)

                            # Normalize Phone 2D coordinates (except FTM range) to be in the same scale in Cam domain
                            seq_in_Phone2_c_ls, pre_Phone2 = [], []
                            for p_i in range(C.recent_K):
                                if p_i == 0: Phone2 = [0, 0]
                                else:
                                    dx = (seq_in_Phone3_c[0,p_i,0] - seq_in_Phone3_c[0,p_i - 1,0]) * scale_in_phone
                                    dy = (seq_in_Phone3_c[0,p_i,1] - seq_in_Phone3_c[0,p_i - 1,1]) * scale_in_phone
                                    Phone2 = [pre_Phone2[0] + dx, pre_Phone2[1] + dy]
                                seq_in_Phone2_c_ls.append(Phone2)
                                pre_Phone2 = Phone2

                            seq_in_Phone3_c[0,:,:2] = np.array(seq_in_Phone2_c_ls)

                        # print(); print() # debug
                        # print('After normalization')
                        # print('seq_in_BBXC3_r: ', seq_in_BBXC3_r)
                        # print('seq_in_Phone3_c: ', seq_in_Phone3_c)
                        # print('np.shape(seq_in_BBXC3_r): ', np.shape(seq_in_BBXC3_r))
                        # print('np.shape(seq_in_Phone3_c): ', np.shape(seq_in_Phone3_c))
                        '''
                        e.g.
                        seq_in_BBXC3_r:  [[[1195.          469.            7.34640169]
                          [1168.          456.            7.86406231]
                          [1148.          449.            8.15638638]
                          [1128.          450.            8.2352066 ]
                          [1099.          449.            9.09870338]
                          [1082.          449.            9.08148861]
                          [1070.          448.            9.35062027]
                          [1055.          439.            9.73178196]
                          [1037.          437.           10.06332111]
                          [1028.          428.            9.8390789 ]]]
                        seq_in_Phone3_c:  [[[    0.             0.             5.587     ]
                          [ -150.52422685   191.77861119     5.587     ]
                          [  482.21748786  -278.8845582      5.587     ]
                          [ -546.58238141  1100.19636846     5.587     ]
                          [  261.06851208  -517.03470554     5.334     ]
                          [  299.0152606    286.05389618     5.334     ]
                          [  -71.18329086  -512.75668466     5.334     ]
                          [ 1149.73515943 -1069.53589908     7.382     ]
                          [  603.04745944   173.83418638     7.382     ]
                          [   49.35155911   164.72529743     7.382     ]]]
                        '''
                        # <<< Phone 2D coordinates Normalization <<<

                        if len(np.unique(seq_in_BBXC3_r)) <= 2:
                            seq_in_BBXC3_r[0,-1,-1] += 0.1 # To avoid "Input matrices must contain >1 unique points" error
                        if len(np.unique(seq_in_Phone3_c)) <= 2:
                            seq_in_Phone3_c[0,-1,-1] += 0.1 # To avoid "Input matrices must contain >1 unique points" error
                        _, _, dist_Cam = procrustes(seq_in_BBXC3_r[0], seq_in_Phone3_c[0]) # prct
                        _, _, dist_Phone = procrustes(seq_in_BBXC3_r[0], seq_in_Phone3_c[0]) # prct

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
                C.scene_eval_stats['gd']['Cam']['IDSWITCH'] = np.nan
                # <<< IDSWITCH <<<

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
                C.scene_eval_stats['gd']['Phone']['IDSWITCH'] = np.nan
                # <<< IDSWITCH <<<

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
                C.scene_eval_stats['hg']['Cam']['IDSWITCH'] = np.nan

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
                C.scene_eval_stats['hg']['Phone']['IDSWITCH'] = np.nan

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

    C.scene_eval_stats_path_to_save = C.checkpoint_path + '/' + \
        C.eval_log_id + '_' + log_str + '_dfv3_scene_eval_stats.pkl'
    # pickle.dump(C.scene_eval_stats_path_to_save, open(C.scene_eval_stats_path_to_save, 'wb'))
    print(C.scene_eval_stats_path_to_save, 'saved!')

# -------
#  Start
# -------
if __name__ == '__main__':
    prepare_testing_data()
    eval_association()
