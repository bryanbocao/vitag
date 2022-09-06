import sys
sys.path.append('../')
from argparse import ArgumentParser
import json
import os
import shutil
import cv2
import numpy as np
import math
from numpy import linalg as LA
from scipy.spatial import distance

import pathlib
import time
import copy
import matplotlib.pyplot as plt
import pickle
import datetime
import json
from collections import defaultdict
from csv import reader
from utils import *
import glob
from PIL import Image
'''
For Passers-by
'''

# TypeError: Object of <some_code> is not JSON serializable
# Ref: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# ----------------------------------------
#  Curations of the whole experiment
# ----------------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        self.user = 'brcao' # 'anon'
        self.root_path = '/media/' + self.user + '/eData2' # '/home/' + self.user
        self.seq_root_path = self.root_path + '/Data/datasets/RAN/seqs/outdoor'
        self.IMU_200 = False # edit
        if self.IMU_200: self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN4model_dfv4_IMU_200'
        else: self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN4model_dfv4'
        self.seq4model_root_path = self.dataset4model_root_path + '/seqs/outdoor'
        if not os.path.exists(self.seq4model_root_path):
            os.makedirs(self.seq4model_root_path)
        print('self.seq_root_path: ', self.seq_root_path)
        # self.scene_ls = ['scene0', 'scene1', 'scene2', 'scene3', 'scene4']
        self.seq_id_path_ls = glob.glob(self.seq_root_path + '/**/*')
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('len(self.seq_id_path_ls): ', len(self.seq_id_path_ls)) # 79
        self.seq_id_ls = [seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls]

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
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)
        self.seq_id = self.seq_id_ls[0] # dummy one
        self.exp = 1 # edit
        self.exp_path = self.root_path + '/Data/datasets/RAN/seqs/exps/exp' + str(self.exp)
        self.meta_path = self.exp_path + '/meta_outdoor.csv'

        if '20210907' in self.seq_id: self.scene_id = 1
        elif '20211004' in self.seq_id: self.scene_id = 2
        elif '20211006' in self.seq_id: self.scene_id = 3
        elif self.seq_id < '20211007_120000': self.scene_id = 4
        elif self.seq_id > '20211007_120000': self.scene_id = 5

        if self.scene_id == 1 or self.scene_id == 2: self.subjects = [43, 80, 0]
        else: self.subjects = [43, 80, 53, 0]
        print(); print() # debug
        print('self.subjects: ', self.subjects)
        self.subj_to_rand_id_file_path = self.exp_path + '/subj_to_rand_id.json'
        with open(self.subj_to_rand_id_file_path, 'r') as f:
            self.subj_to_id_dict = json.load(f)
            print(self.subj_to_rand_id_file_path, 'loaded!')

        self.seq_path = self.seq_root_path + '/scene' + str(self.scene_id) + '/' + self.seq_id
        self.seq_date = self.seq_id[:8]
        self.seq_id_to_start_end_ots_offsets = defaultdict()
        self.subj_to_offset = defaultdict()
        self.start_ots, self.end_ots, self.subj_to_offset = get_start_end_ts_update_phone_with_offsets(self.seq_id, self.meta_path)
        self.seq_path_for_model = self.seq4model_root_path + '/scene' + str(self.scene_id) + '/' + self.seq_id
        self.img_type = 'Depth'
        self.img_path = self.seq_path + '/' + self.img_type

        # -----
        #  GND
        # -----
        self.GND_path = self.seq_path + '/GND/RAN_' + self.seq_id + '-export.json'
        self.GND_ts16_dfv4_path = self.seq_path + '/GND_ts16_dfv4'
        if not os.path.exists(self.GND_ts16_dfv4_path): os.makedirs(self.GND_ts16_dfv4_path)
        self.GND_ts16_dfv4_to_BBX5_dfv4_path = self.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX5_dfv4.json'
        with open(self.GND_ts16_dfv4_to_BBX5_dfv4_path, 'r') as f:
            self.GND_ts16_dfv4_to_BBX5_dfv4 = json.load(f)
            print(self.GND_ts16_dfv4_to_BBX5_dfv4_path, 'loaded!')

        self.max_depth = 500

        # -----
        #  IMU
        # -----
        self.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMUlgyq10_data_types = ['LINEAR', 'GYRO', 'Quaternion']
        self.IMUlgyqm13_data_types = ['LINEAR', 'GYRO', 'Quaternion', 'MAG']
        self.IMUaccgym_data_types = ['ACCEL', 'GYRO', 'MAG']
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv4_path = self.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.IMU_dfv4_path): os.makedirs(self.IMU_dfv4_path)
        self.subj_id_to_IMU19_ts16_dfv4_path = ''
        self.subj_id_to_IMU19_ts16_dfv4 = defaultdict()
        self.subj_id_to_IMUlgyq10_ts16_dfv4_path = ''
        self.subj_id_to_IMUlgyq10_ts16_dfv4 = defaultdict()
        self.subj_id_to_IMUlgyqm13_ts16_dfv4_path = ''
        self.subj_id_to_IMUlgyqm13_ts16_dfv4 = defaultdict()
        # self.subj_id_to_IMU_NED_pos_ts16_dfv4_path = '' # to be added
        # self.subj_id_to_IMU_NED_pos_ts16_dfv4 = defaultdict() # to be added

        self.subj_id_to_IMU19_200_ts16_dfv4_path = ''
        self.subj_id_to_IMU19_200_ts16_dfv4 = defaultdict()
        self.subj_id_to_IMUaccgym_200_ts16_dfv4_path = ''
        self.subj_id_to_IMUaccgym_200_ts16_dfv4 = defaultdict()

        # ---------------------------------------------------
        #  Synchronized data: BBXC3,BBX5,Others_dfv4 to save
        # ---------------------------------------------------
        self.BBXC3_Others_sync_dfv4, self.BBX5_Others_sync_dfv4 = [], []
        self.sync_dfv4_path = self.seq_path_for_model + '/sync_ts16_dfv4'
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)

        self.Others_id_ls = []
        self.Others_id_ls_path = self.sync_dfv4_path + '/Others_id_ls.pkl'

        self.BBXC3_Others_sync_dfv4_path = self.sync_dfv4_path + '/BBXC3_Others_sync_dfv4.pkl'
        self.BBX5_Others_sync_dfv4_path = self.sync_dfv4_path + '/BBX5_Others_sync_dfv4.pkl'
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)
        #  To de updated in update_parameters()
        # --------------------------------------

C = Config()

def update_parameters(seq_id_idx):
    C.seq_id = C.seq_id_ls[seq_id_idx]
    if '20210907' in C.seq_id: C.scene_id = 1
    elif '20211004' in C.seq_id: C.scene_id = 2
    elif '20211006' in C.seq_id: C.scene_id = 3
    elif C.seq_id < '20211007_120000': C.scene_id = 4
    elif C.seq_id > '20211007_120000': C.scene_id = 5
    print(); print() # debug
    print('C.scene: ', C.scene_id)

    if C.scene_id == 1 or C.scene_id == 2: C.subjects = [43, 80]
    else: C.subjects = [43, 80, 53]
    print(); print() # debug
    print('C.subjects: ', C.subjects)
    C.subj_to_rand_id_file_path = C.exp_path + '/subj_to_rand_id.json'
    with open(C.subj_to_rand_id_file_path, 'r') as f:
        C.subj_to_id_dict = json.load(f)
        print(C.subj_to_rand_id_file_path, 'loaded!')
    C.seq_path = C.seq_root_path + '/scene' + str(C.scene_id) + '/' + C.seq_id
    C.seq_date = C.seq_id[:8]
    C.start_ots, C.end_ots, C.subj_to_offset = get_start_end_ts_update_phone_with_offsets(C.seq_id, C.meta_path)
    C.seq_path_for_model = C.seq4model_root_path + '/scene' + str(C.scene_id) + '/' + C.seq_id
    if not os.path.exists(C.seq_path_for_model): os.makedirs(C.seq_path_for_model)
    C.img_path = C.seq_path + '/' + C.img_type
    C.RGB_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
    with open(C.RGB_ts16_dfv4_ls_path, 'r') as f:
        C.RGB_ts16_dfv4_ls = json.load(f)
        print(C.RGB_ts16_dfv4_ls_path, 'loaded!')

    print('\n\nProcessing seq {}/{}'.format(seq_id_idx + 1, len(C.seq_id_ls)))

    # -----
    #  GND
    # -----
    C.GND_path = C.seq_path + '/GND/RAN_' + C.seq_id + '-export.json'
    C.GND_ts16_dfv4_path = C.seq_path + '/GND_ts16_dfv4'
    if not os.path.exists(C.GND_ts16_dfv4_path): os.makedirs(C.GND_ts16_dfv4_path)
    C.GND_ts16_dfv4_to_BBX5_dfv4_path = C.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX5_dfv4.json'
    with open(C.GND_ts16_dfv4_to_BBX5_dfv4_path, 'r') as f:
        C.GND_ts16_dfv4_to_BBX5_dfv4 = json.load(f)
        print(C.GND_ts16_dfv4_to_BBX5_dfv4_path, 'loaded!')

    # ---------------------------------------------------
    #  Synchronized data: BBXC3,BBX5,Others_dfv4 to save
    # ---------------------------------------------------
    C.BBXC3_Others_sync_dfv4, C.BBX5_Others_sync_dfv4 = [], []
    C.sync_dfv4_path = C.seq_path_for_model + '/sync_ts16_dfv4'
    if not os.path.exists(C.sync_dfv4_path): os.makedirs(C.sync_dfv4_path)

    C.Others_id_ls = []
    C.Others_id_ls_path = C.sync_dfv4_path + '/Others_id_ls.pkl'

    C.BBXC3_Others_sync_dfv4_path = C.sync_dfv4_path + '/BBXC3_Others_sync_dfv4.pkl'
    C.BBX5_Others_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_Others_sync_dfv4.pkl'
    if not os.path.exists(C.sync_dfv4_path): os.makedirs(C.sync_dfv4_path)

def process_save_others_sync_dfv4():
    # ----------------
    #  Get all Others
    # ----------------
    Others_id_set = set()
    for i, ts16_dfv4 in enumerate(C.RGB_ts16_dfv4_ls):
        # -----------------------
        #  Valid Timestamp range
        # -----------------------
        if ts16_dfv4 >= ots26_to_ts16_dfv4(C.start_ots) and ts16_dfv4 <= ots26_to_ts16_dfv4(C.end_ots):
            BBXC3_ts16_dfv4_ls, BBX5_ts16_dfv4_ls = [], []
            # print(); print() # debug
            # print('ts16_dfv4: ', ts16_dfv4)
            # e.g. 1608750784.030037

            # ------------
            #  Passers-by
            # ------------
            # Only process data for passers-by
            subj = C.subjects[-1]
            if ts16_dfv4 in C.GND_ts16_dfv4_to_BBX5_dfv4:
                subjs_ts16_dfv4 =  C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4].keys()
                for subj_ in subjs_ts16_dfv4:
                    if 'Other' in subj_:
                        # print('\n\nsubjs_ts16_dfv4: ', subjs_ts16_dfv4)
                        Others_id_set.add(subj_)
    C.Others_id_ls = sorted(list(Others_id_set))
    # print('C.Others_id_ls: ', C.Others_id_ls)

    # ----------------
    #  Each Timestamp
    # ----------------
    for i, ts16_dfv4 in enumerate(C.RGB_ts16_dfv4_ls):
        # print(); print() # debug
        # print('i: ', i, ', ts16_dfv4: ', ts16_dfv4)
        # print('len(C.RGB_ts16_dfv4_ls): ', len(C.RGB_ts16_dfv4_ls))
        # print('ots26_to_ts16_dfv4(C.start_ots): ', ots26_to_ts16_dfv4(C.start_ots))
        # print('ots26_to_ts16_dfv4(C.end_ots): ', ots26_to_ts16_dfv4(C.end_ots))
        # -----------------------
        #  Valid Timestamp range
        # -----------------------
        if ts16_dfv4 >= ots26_to_ts16_dfv4(C.start_ots) and ts16_dfv4 <= ots26_to_ts16_dfv4(C.end_ots):
            BBXC3_ts16_dfv4_ls, BBX5_ts16_dfv4_ls = [], []
            # print(); print() # debug
            # print('ts16_dfv4: ', ts16_dfv4)
            # e.g. 1608750784.030037

            # ------------
            #  Passers-by
            # ------------
            # -----
            #  BBX
            # -----
            if ts16_dfv4 in C.GND_ts16_dfv4_to_BBX5_dfv4:
                subjs_ts16_dfv4 =  C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4].keys()
                for subj_ in C.Others_id_ls:
                    # print(); print() # debug
                    # print('subj_: ', subj_)
                    # e.g. {'subj': 'Other19', 'c_row': 253, 'c_col': 558, 'w': 36, 'h': 103, 'd': 9.784972190856934}
                    BBXC3_ts16_dfv4_subj = [[np.nan, np.nan, np.nan]]
                    BBX5_ts16_dfv4_subj = [[np.nan, np.nan, np.nan, np.nan, np.nan]]
                    if subj_ in subjs_ts16_dfv4:
                        # print(); print() # debug
                        # print('C.GND_ts16_dfv4_to_BBX5_dfv4: ', C.GND_ts16_dfv4_to_BBX5_dfv4)
                        # print('C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4]:', C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4])

                        d = C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj_]
                        # print(); print() # debug
                        # print('C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj]: ', C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj_])
                        # e.g. {'subj': 'Other29', 'c_row': 234, 'c_col': 695, 'w': 34, 'h': 94, 'd': 10.807808876037598}
                        if d['d'] < C.max_depth:
                            BBXC3_ts16_dfv4_subj = [[d['c_col'], d['c_row'], d['d']]]
                            BBX5_ts16_dfv4_subj = [[d['c_col'], d['c_row'], d['d'], d['w'], d['h']]]
                    BBXC3_ts16_dfv4_ls.append(BBXC3_ts16_dfv4_subj)
                    BBX5_ts16_dfv4_ls.append(BBX5_ts16_dfv4_subj)

                # print('np.shape(BBXC3_ts16_dfv4_ls): ', np.shape(BBXC3_ts16_dfv4_ls))
                # print('np.shape(BBX5_ts16_dfv4_ls): ', np.shape(BBX5_ts16_dfv4_ls))

                # ----------------------------
                #  Save data in one ts16_dfv4
                # ----------------------------
                C.BBXC3_Others_sync_dfv4.append(BBXC3_ts16_dfv4_ls)
                C.BBX5_Others_sync_dfv4.append(BBX5_ts16_dfv4_ls)

    C.BBXC3_Others_sync_dfv4 = np.array(C.BBXC3_Others_sync_dfv4)
    C.BBX5_Others_sync_dfv4 = np.array(C.BBX5_Others_sync_dfv4)
    # --------
    #  Verify
    # --------
    print('np.shape(C.BBXC3_Others_sync_dfv4): ', np.shape(C.BBXC3_Others_sync_dfv4))
    print('np.shape(C.BBX5_Others_sync_dfv4): ', np.shape(C.BBX5_Others_sync_dfv4))
    '''
    e.g.
    202012 RGB: 3fps
    202109, 202110 RGB: 10fps
    np.shape(C.BBXC3_Others_sync_dfv4):  (1769, 38, 1, 3)
    np.shape(C.BBX5_Others_sync_dfv4):  (1769, 38, 1, 5)
    '''

    # ----------------
    #  Save Sync Data
    # ----------------
    pickle.dump(C.Others_id_ls, open(C.Others_id_ls_path, 'wb'))
    pickle.dump(C.BBXC3_Others_sync_dfv4, open(C.BBXC3_Others_sync_dfv4_path, 'wb'))
    pickle.dump(C.BBX5_Others_sync_dfv4, open(C.BBX5_Others_sync_dfv4_path, 'wb'))
    print(C.BBXC3_Others_sync_dfv4_path, 'saved!'); print(C.BBX5_Others_sync_dfv4_path, 'saved!')


# ------------------
#  Process all seqs
# ------------------
# ---------
#  Outdoor
# ---------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    process_save_others_sync_dfv4()
