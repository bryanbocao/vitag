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
        self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN4model_dfv4'
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
        self.exp = 1 # edit
        self.seq_id_ls = [seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls]
        self.seq_id_to_start_end_ts16_dfv4 = defaultdict()
        self.start_end_ts16_indoor_dfv4_path = self.dataset4model_root_path + '/exps/exp' + \
            str(self.exp) + '/start_end_ts16_dfv4_outdoor.json'

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
        self.img_type = 'RGB_ts16_dfv4_anonymized'
        self.img_path = self.seq_path + '/' + self.img_type
        self.RGB_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
        with open(self.RGB_ts16_dfv4_ls_path, 'r') as f:
            self.RGB_ts16_dfv4_ls = json.load(f)
            print(self.RGB_ts16_dfv4_ls_path, 'loaded!')

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

        # -----
        #  IMU
        # -----
        self.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMUlgyq10_data_types = ['LINEAR', 'GYRO', 'Quaternion']
        self.IMUlgyqm13_data_types = ['LINEAR', 'GYRO', 'Quaternion', 'MAG']
        self.IMUaccgym9_data_types = ['ACCEL', 'GYRO', 'MAG']
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

        if self.IMU_200:
            self.subj_id_to_IMU19_200_ts16_dfv4_path = ''
            self.subj_id_to_IMU19_200_ts16_dfv4 = defaultdict()
            self.subj_id_to_IMUaccgym9_200_ts16_dfv4_path = ''
            self.subj_id_to_IMUaccgym9_200_ts16_dfv4 = defaultdict()

        # -----
        #  FTM
        # -----
        self.FTM_path = self.seq_path + '/WiFi'
        self.FTM_dfv4_path = self.seq_path + '/FTM_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.FTM_dfv4_path): os.makedirs(self.FTM_dfv4_path)
        self.subj_id_to_FTM_ts16_dfv4_path = ''
        self.subj_id_to_FTM_ts16_dfv4 = defaultdict()

        # ----------------------------------------------------------
        #  Synchronized data: BBXC3,BBX5,FTM,IMU,_sync_dfv4 to save
        # ----------------------------------------------------------
        # One sample per frame >>>
        self.BBXC3_sync_dfv4, self.BBX5_sync_dfv4 = [], []
        self.IMU19_sync_dfv4, self.IMUlgyq10_sync_dfv4 = [], []
        # self.IMU_NED_pos_sync_dfv4 = [] # to be added
        self.IMUlgyqm13_sync_dfv4 = []
        self.FTM_sync_dfv4 = []
        self.seq_id = self.seq_id_ls[0]
        self.RGB_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
        self.RGBg_ts16_dfv4_ls = []
        self.RGBg_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'
        self.sync_dfv4_path = self.seq_path_for_model + '/sync_ts16_dfv4'
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)
        self.BBXC3_sync_dfv4_path = self.sync_dfv4_path + '/BBXC3_sync_dfv4.pkl'
        self.BBX5_sync_dfv4_path = self.sync_dfv4_path + '/BBX5_sync_dfv4.pkl'
        self.IMU19_sync_dfv4_path = self.sync_dfv4_path + '/IMU19_sync_dfv4.pkl'
        self.IMUlgyq10_sync_dfv4_path = self.sync_dfv4_path + '/IMUlgyq10_sync_dfv4.pkl'
        self.IMUlgyqm13_sync_dfv4_path = self.sync_dfv4_path + '/IMUlgyqm13_sync_dfv4.pkl'
        # self.IMU_NED_pos_sync_dfv4_path = self.sync_dfv4_path + '/IMU_NED_pos_sync_dfv4.pkl' # to be added
        self.FTM_sync_dfv4_path = self.sync_dfv4_path + '/FTM_sync_dfv4.pkl'
        if not os.path.exists(self.sync_dfv4_path): os.makedirs(self.sync_dfv4_path)
        # One sample per frame <<<

        if self.IMU_200:
            # 200 samples per frame >>>
            self.IMU19_200_sync_dfv4, self.IMUaccgym9_200_sync_dfv4 = [[] for _ in range(200)], [[] for _ in range(200)]
            self.IMU19_200_sync_dfv4_path = self.sync_dfv4_path + '/IMU19_200_sync_dfv4.pkl'
            self.IMUaccgym9_200_sync_dfv4_path = self.sync_dfv4_path + '/IMUaccgym9_200_sync_dfv4.pkl'
            # 200 samples per frame <<<
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
    C.seq_id_to_start_end_ts16_dfv4[C.seq_id] =  defaultdict()
    C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start'] = ots26_to_ts16_dfv4(C.start_ots)
    C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end'] = ots26_to_ts16_dfv4(C.end_ots)

    C.seq_path_for_model = C.seq4model_root_path + '/scene' + str(C.scene_id) + '/' + C.seq_id
    if not os.path.exists(C.seq_path_for_model): os.makedirs(C.seq_path_for_model)
    C.img_path = C.seq_path + '/' + C.img_type
    C.RGB_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
    with open(C.RGB_ts16_dfv4_ls_path, 'r') as f:
        C.RGB_ts16_dfv4_ls = json.load(f)
        print(C.RGB_ts16_dfv4_ls_path, 'loaded!')
    C.RGBg_ts16_dfv4_ls = []
    C.RGBg_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBg_ts16_dfv4_ls.json'

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

    # -----
    #  IMU
    # -----
    C.IMU19_data_types = ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
    C.IMUlgyq10_data_types = ['LINEAR', 'GYRO', 'Quaternion']
    C.IMU_path = C.seq_path + '/IMU'
    C.IMU_dfv4_path = C.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
    if not os.path.exists(C.IMU_dfv4_path): os.makedirs(C.IMU_dfv4_path)
    C.subj_id_to_IMU19_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_ts16_dfv4.json'
    with open(C.subj_id_to_IMU19_ts16_dfv4_path, 'r') as f:
        C.subj_id_to_IMU19_ts16_dfv4 = json.load(f)
        print(C.subj_id_to_IMU19_ts16_dfv4_path, 'loaded!')
    C.subj_id_to_IMUlgyq10_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyq10_ts16_dfv4.json'
    with open(C.subj_id_to_IMUlgyq10_ts16_dfv4_path, 'r') as f:
        C.subj_id_to_IMUlgyq10_ts16_dfv4 = json.load(f)
        print(C.subj_id_to_IMUlgyq10_ts16_dfv4_path, 'loaded!')
    C.subj_id_to_IMUlgyqm13_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyqm13_ts16_dfv4.json'
    with open(C.subj_id_to_IMUlgyqm13_ts16_dfv4_path, 'r') as f:
        C.subj_id_to_IMUlgyqm13_ts16_dfv4 = json.load(f)
        print(C.subj_id_to_IMUlgyqm13_ts16_dfv4_path, 'loaded!')

    # to be added
    # C.subj_id_to_IMU_NED_pos_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU_NED_pos_ts16_dfv4.json'
    # with open(C.subj_id_to_IMU_NED_pos_ts16_dfv4_path, 'r') as f:
    #     C.subj_id_to_IMU_NED_pos_ts16_dfv4 = json.load(f)
    #     print(C.subj_id_to_IMU_NED_pos_ts16_dfv4_path, 'loaded!')

    if C.IMU_200:
        # 200 samples per frame >>>
        C.subj_id_to_IMU19_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_200_ts16_dfv4.json'
        with open(C.subj_id_to_IMU19_200_ts16_dfv4_path, 'r') as f:
            C.subj_id_to_IMU19_200_ts16_dfv4 = json.load(f)
            print(C.subj_id_to_IMU19_200_ts16_dfv4_path, 'loaded!')
        C.subj_id_to_IMUaccgym9_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUaccgym9_200_ts16_dfv4.json'
        with open(C.subj_id_to_IMUaccgym9_200_ts16_dfv4_path, 'r') as f:
            C.subj_id_to_IMUaccgym9_200_ts16_dfv4 = json.load(f)
            print(C.subj_id_to_IMUaccgym9_200_ts16_dfv4_path, 'loaded!')
        # 200 samples per frame <<<

    # -----
    #  FTM
    # -----
    C.FTM_path = C.seq_path + '/WiFi'
    C.FTM_dfv4_path = C.seq_path + '/FTM_dfv4' # ts13_dfv4 with offsets (os)
    if not os.path.exists(C.FTM_dfv4_path): os.makedirs(C.FTM_dfv4_path)
    C.subj_id_to_FTM_ts16_dfv4_path = C.FTM_dfv4_path + '/subj_id_to_FTM_ts16_dfv4.json'
    with open(C.subj_id_to_FTM_ts16_dfv4_path, 'r') as f:
        C.subj_id_to_FTM_ts16_dfv4 = json.load(f)
        print(C.subj_id_to_FTM_ts16_dfv4_path, 'loaded!')

    # ------------------------------------------------------
    #  Synchronized data: BBXC3,BBX5,IMU,_sync_dfv4 to save
    # ------------------------------------------------------
    C.BBXC3_sync_dfv4, C.BBX5_sync_dfv4 = [], []
    C.IMU19_sync_dfv4, C.IMUlgyq10_sync_dfv4 = [], []
    C.IMUlgyqm13_sync_dfv4 = []
    # C.IMU_NED_pos_sync_dfv4 = [] # to be added
    C.FTM_sync_dfv4 = []
    if C.IMU_200:
        # 200 samples per frame >>>
        C.IMU19_200_sync_dfv4, C.IMUaccgym9_200_sync_dfv4 = [], []
        # 200 samples per frame <<<

    C.sync_dfv4_path = C.seq_path_for_model + '/sync_ts16_dfv4'
    if not os.path.exists(C.sync_dfv4_path): os.makedirs(C.sync_dfv4_path)
    C.BBXC3_sync_dfv4_path = C.sync_dfv4_path + '/BBXC3_sync_dfv4.pkl'
    C.BBX5_sync_dfv4_path = C.sync_dfv4_path + '/BBX5_sync_dfv4.pkl'
    C.IMU19_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_sync_dfv4.pkl'
    C.IMUlgyq10_sync_dfv4_path = C.sync_dfv4_path + '/IMUlgyq10_sync_dfv4.pkl'
    C.IMUlgyqm13_sync_dfv4_path = C.sync_dfv4_path + '/IMUlgyqm13_sync_dfv4.pkl'
    # C.IMU_NED_pos_sync_dfv4_path = C.sync_dfv4_path + '/IMU_NED_pos_sync_dfv4.pkl' # to be added
    C.FTM_sync_dfv4_path = C.sync_dfv4_path + '/FTM_sync_dfv4.pkl'
    if C.IMU_200:
        # 200 samples per frame >>>
        C.IMU19_200_sync_dfv4_path = C.sync_dfv4_path + '/IMU19_200_sync_dfv4.pkl'
        C.IMUaccgym9_200_sync_dfv4_path = C.sync_dfv4_path + '/IMUaccgym9_200_sync_dfv4.pkl'
        # 200 samples per frame <<<
    if not os.path.exists(C.sync_dfv4_path): os.makedirs(C.sync_dfv4_path)

def process_save_sync_dfv4():
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
            C.RGBg_ts16_dfv4_ls.append(ts16_dfv4)
            BBXC3_ts16_dfv4_ls, BBX5_ts16_dfv4_ls = [], []
            IMU19_ts16_dfv4_ls, IMUlgyq10_ts16_dfv4_ls = [], []
            IMUlgyqm13_ts16_dfv4_ls = []
            # IMU_NED_pos_ts16_dfv4_ls = [] # to be added
            if C.IMU_200:
                # 200 samples per frame >>>
                IMU19_200_ts16_dfv4_ls, IMUaccgym9_200_ts16_dfv4_ls = [], []
                # 200 samples per frame <<<
            FTM_ts16_dfv4_ls = []
            # print(); print() # debug
            # print('ts16_dfv4: ', ts16_dfv4)
            # e.g. 1608750784.030037

            # --------------
            #  Each Subject
            # --------------
            for subj_i, subj in enumerate(C.subjects):
                if subj == 'Others': continue # Only process data with phone data
                subj_id = str(C.subj_to_id_dict['Outdoor'][subj])
                # -----
                #  BBX
                # -----
                BBXC3_ts16_dfv4_subj = [[np.nan, np.nan, np.nan]]
                BBX5_ts16_dfv4_subj = [[np.nan, np.nan, np.nan, np.nan, np.nan]]
                if ts16_dfv4 in C.GND_ts16_dfv4_to_BBX5_dfv4 and subj in C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4]:
                    # print(); print() # debug
                    # print('C.GND_ts16_dfv4_to_BBX5_dfv4: ', C.GND_ts16_dfv4_to_BBX5_dfv4)
                    # print('C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4]:', C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4])

                    d = C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj]
                    # print(); print() # debug
                    # print('C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj]: ', C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj])
                    BBXC3_ts16_dfv4_subj = [[d['c_col'], d['c_row'], d['d']]]
                    BBX5_ts16_dfv4_subj = [[d['c_col'], d['c_row'], d['d'], d['w'], d['h']]]
                BBXC3_ts16_dfv4_ls.append(BBXC3_ts16_dfv4_subj)
                BBX5_ts16_dfv4_ls.append(BBX5_ts16_dfv4_subj)

                # -----
                #  IMU
                # -----
                # print('\n\n C.subj_id_to_IMU19_ts16_dfv4: ', C.subj_id_to_IMU19_ts16_dfv4)
                # print('\n\n C.subj_id_to_IMU19_ts16_dfv4[subj_id]: ', C.subj_id_to_IMU19_ts16_dfv4[subj_id])
                IMU19_ts16_dfv4_subj = C.subj_id_to_IMU19_ts16_dfv4[subj_id][ts16_dfv4]
                # print(); print() # debug
                # print('IMU19_ts16_dfv4_subj: ', IMU19_ts16_dfv4_subj)
                IMU19_ts16_dfv4_ls.append(IMU19_ts16_dfv4_subj)

                IMUlgyq10_ts16_dfv4_subj = C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][ts16_dfv4]
                # print(); print() # debug
                # print('IMUlgyq10_ts16_dfv4_subj: ', IMUlgyq10_ts16_dfv4_subj)
                IMUlgyq10_ts16_dfv4_ls.append(IMUlgyq10_ts16_dfv4_subj)

                IMUlgyqm13_ts16_dfv4_subj = C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][ts16_dfv4]
                # print(); print() # debug
                # print('IMUlgyqm13_ts16_dfv4_subj: ', IMUlgyqm13_ts16_dfv4_subj)
                IMUlgyqm13_ts16_dfv4_ls.append(IMUlgyqm13_ts16_dfv4_subj)

                # to be added
                # IMU_NED_pos_ts16_dfv4_subj = C.subj_id_to_IMU_NED_pos_ts16_dfv4[subj_id][ts16_dfv4]
                # print(); print() # debug
                # print('IMU_NED_pos_ts16_dfv4_subj: ', IMU_NED_pos_ts16_dfv4_subj)
                # IMU_NED_pos_ts16_dfv4_ls.append(IMU_NED_pos_ts16_dfv4_subj)

                if C.IMU_200:
                    # 200 samples per frame >>>
                    IMU19_200_ts16_dfv4_subj = C.subj_id_to_IMU19_200_ts16_dfv4[subj_id][ts16_dfv4]
                    # print(); print() # debug
                    # print('np.shape(IMU19_200_ts16_dfv4_subj): ', np.shape(IMU19_200_ts16_dfv4_subj))
                    IMU19_200_ts16_dfv4_ls.append(IMU19_200_ts16_dfv4_subj)

                    IMUaccgym9_200_ts16_dfv4_subj = C.subj_id_to_IMUaccgym9_200_ts16_dfv4[subj_id][ts16_dfv4]
                    # print(); print() # debug
                    # print('np.shape(IMUaccgym9_200_ts16_dfv4_subj): ', np.shape(IMUaccgym9_200_ts16_dfv4_subj))
                    IMUaccgym9_200_ts16_dfv4_ls.append(IMUaccgym9_200_ts16_dfv4_subj)
                    # 200 samples per frame <<<

                # -----
                #  FTM
                # -----
                # print(list(C.subj_id_to_FTM_ts16_dfv4[subj_id].keys())[0])
                # print(list(C.subj_id_to_FTM_ts16_dfv4[subj_id].keys())[-1])
                FTM_ts16_dfv4_subj = C.subj_id_to_FTM_ts16_dfv4[subj_id][ts16_dfv4]
                # print(); print() # debug
                # print('FTM_ts16_dfv4_subj: ', FTM_ts16_dfv4_subj)
                FTM_ts16_dfv4_ls.append(FTM_ts16_dfv4_subj)

            # print('np.shape(BBXC3_ts16_dfv4_ls): ', np.shape(BBXC3_ts16_dfv4_ls))
            # print('np.shape(BBX5_ts16_dfv4_ls): ', np.shape(BBX5_ts16_dfv4_ls))
            # print('np.shape(IMU19_ts16_dfv4_ls): ', np.shape(IMU19_ts16_dfv4_ls))
            if C.IMU_200:
                print('np.shape(IMU19_200_ts16_dfv4_ls): ', np.shape(IMU19_200_ts16_dfv4_ls))
                print('np.shape(IMUaccgym9_200_ts16_dfv4_ls): ', np.shape(IMUaccgym9_200_ts16_dfv4_ls))
                print('np.shape(FTM_ts16_dfv4_ls): ', np.shape(FTM_ts16_dfv4_ls))
            '''
            e.g.
            np.shape(BBXC3_ts16_dfv4_ls):  (3, 1, 3)
            np.shape(BBX5_ts16_dfv4_ls):  (3, 1, 5)
            np.shape(IMU19_ts16_dfv4_ls):  (3, 1, 19)
            np.shape(IMU19_200_ts16_dfv4_ls):  (3, 200, 19)
            np.shape(IMUaccgym9_200_ts16_dfv4_ls):  (3, 200, 9)
            np.shape(FTM_ts16_dfv4_ls):  (3, 1, 2) #
            '''

            # ----------------------------
            #  Save data in one ts16_dfv4
            # ----------------------------
            C.BBXC3_sync_dfv4.append(BBXC3_ts16_dfv4_ls)
            C.BBX5_sync_dfv4.append(BBX5_ts16_dfv4_ls)
            C.IMU19_sync_dfv4.append(IMU19_ts16_dfv4_ls)
            C.IMUlgyq10_sync_dfv4.append(IMUlgyq10_ts16_dfv4_ls)
            C.IMUlgyqm13_sync_dfv4.append(IMUlgyqm13_ts16_dfv4_ls)
            if C.IMU_200:
                # 200 samples per frame >>>
                C.IMU19_200_sync_dfv4.append(IMU19_200_ts16_dfv4_ls)
                C.IMUaccgym9_200_sync_dfv4.append(IMUaccgym9_200_ts16_dfv4_ls)
                # 200 samples per frame <<<
            # C.IMU_NED_pos_sync_dfv4.append(IMU_NED_pos_ts16_dfv4_ls) # to be added
            C.FTM_sync_dfv4.append(FTM_ts16_dfv4_ls)

    C.BBXC3_sync_dfv4 = np.array(C.BBXC3_sync_dfv4)
    C.BBX5_sync_dfv4 = np.array(C.BBX5_sync_dfv4)
    C.IMU19_sync_dfv4 = np.array(C.IMU19_sync_dfv4)
    C.IMUlgyq10_sync_dfv4 = np.array(C.IMUlgyq10_sync_dfv4)
    C.IMUlgyqm13_sync_dfv4 = np.array(C.IMUlgyqm13_sync_dfv4)
    # C.IMU_NED_pos_sync_dfv4 = np.array(C.IMU_NED_pos_sync_dfv4) # to be added
    if C.IMU_200:
        # 200 samples per frame >>>
        C.IMU19_200_sync_dfv4 = np.array(C.IMU19_200_sync_dfv4)
        C.IMUaccgym9_200_sync_dfv4 = np.array(C.IMUaccgym9_200_sync_dfv4)
        # 200 samples per frame <<<
    C.FTM_sync_dfv4 = np.array(C.FTM_sync_dfv4)
    # --------
    #  Verify
    # --------
    print('np.shape(C.BBXC3_sync_dfv4): ', np.shape(C.BBXC3_sync_dfv4))
    print('np.shape(C.BBX5_sync_dfv4): ', np.shape(C.BBX5_sync_dfv4))
    print('np.shape(C.IMU19_sync_dfv4): ', np.shape(C.IMU19_sync_dfv4))
    print('np.shape(C.IMUlgyq10_sync_dfv4): ', np.shape(C.IMUlgyq10_sync_dfv4))
    print('np.shape(C.IMUlgyqm13_sync_dfv4): ', np.shape(C.IMUlgyqm13_sync_dfv4))
    # print('np.shape(C.IMU_NED_pos_sync_dfv4): ', np.shape(C.IMU_NED_pos_sync_dfv4)) # to be added
    print('np.shape(C.FTM_sync_dfv4): ', np.shape(C.FTM_sync_dfv4))
    if C.IMU_200:
        print('np.shape(C.IMU19_200_sync_dfv4): ', np.shape(C.IMU19_200_sync_dfv4)) # 200 samples per frame
        print('np.shape(C.IMUaccgym9_200_sync_dfv4): ', np.shape(C.IMUaccgym9_200_sync_dfv4)) # 200 samples per frame
    '''
    e.g.
    202012 RGBh: 3fps
    np.shape(C.BBXC3_sync_dfv4):  (554, 5, 1, 3)
    np.shape(C.BBX5_sync_dfv4):  (554, 5, 1, 5)
    np.shape(C.IMU19_sync_dfv4):  (554, 5, 1, 19)
    np.shape(C.IMUlgyq10_sync_dfv4):  (554, 5, 1, 10)
    np.shape(C.IMUlgyqm13_sync_dfv4):  (554, 5, 1, 13)
    np.shape(C.IMU_NED_pos_sync_dfv4):  (554, 5, 1, 2)
    np.shape(C.FTM_sync_dfv4):  (554, 5, 1, 2)
    np.shape(C.IMU19_200_sync_dfv4):  (554, 5, 200, 19)
    np.shape(C.IMUaccgym9_200_sync_dfv4):  (554, 5, 200, 9)
    '''
    '''
    e.g.
    202012 RGB: 3fps
    np.shape(C.BBXC3_sync_dfv4):  (539, 5, 1, 3)
    np.shape(C.BBX5_sync_dfv4):  (539, 5, 1, 5)
    np.shape(C.IMU19_sync_dfv4):  (539, 5, 1, 19)
    202109, 202110 RGB: 10fps
    np.shape(C.BBXC3_sync_dfv4):  (1767, 3, 1, 3)
    np.shape(C.BBX5_sync_dfv4):  (1767, 3, 1, 5)
    np.shape(C.IMU19_sync_dfv4):  (1767, 3, 1, 19)
    np.shape(C.IMUlgyq10_sync_dfv4):  (1767, 3, 1, 10)
    np.shape(C.IMUlgyqm13_sync_dfv4):   (1767, 3, 1, 13)
    '''

    # ----------------
    #  Save Sync Data
    # ----------------
    pickle.dump(C.BBXC3_sync_dfv4, open(C.BBXC3_sync_dfv4_path, 'wb'))
    pickle.dump(C.BBX5_sync_dfv4, open(C.BBX5_sync_dfv4_path, 'wb'))
    pickle.dump(C.IMU19_sync_dfv4, open(C.IMU19_sync_dfv4_path, 'wb'))
    pickle.dump(C.IMUlgyq10_sync_dfv4, open(C.IMUlgyq10_sync_dfv4_path, 'wb'))
    pickle.dump(C.IMUlgyqm13_sync_dfv4, open(C.IMUlgyqm13_sync_dfv4_path, 'wb'))
    # pickle.dump(C.IMU_NED_pos_sync_dfv4, open(C.IMU_NED_pos_sync_dfv4_path, 'wb')) # to be added
    if C.IMU_200:
        pickle.dump(C.IMU19_200_sync_dfv4, open(C.IMU19_200_sync_dfv4_path, 'wb')) # 200 samples per frame
        pickle.dump(C.IMUaccgym9_200_sync_dfv4, open(C.IMUaccgym9_200_sync_dfv4_path, 'wb')) # 200 samples per frame
    pickle.dump(C.FTM_sync_dfv4, open(C.FTM_sync_dfv4_path, 'wb'))
    print(C.BBXC3_sync_dfv4_path, 'saved!'); print(C.BBX5_sync_dfv4_path, 'saved!')
    print(C.IMU19_sync_dfv4_path, 'saved!'); print(C.IMUlgyq10_sync_dfv4_path, 'saved!')
    print(C.IMUlgyqm13_sync_dfv4_path, 'saved!');
    # print(C.IMU_NED_pos_sync_dfv4_path, 'saved!') # to be added
    if C.IMU_200:
        # 200 samples per frame >>>
        print(C.IMU19_200_sync_dfv4_path, 'saved!'); print(C.IMUaccgym9_200_sync_dfv4_path, 'saved!')
        # 200 samples per frame <<<
    print(C.FTM_sync_dfv4_path, 'saved!')

    with open(C.RGBg_ts16_dfv4_ls_path, 'w') as f:
        json.dump(C.RGBg_ts16_dfv4_ls, f, cls=NpEncoder)
        print(C.RGBg_ts16_dfv4_ls_path, 'saved!')

def save_start_end_ts16_indoor_dfv4():
    with open(C.start_end_ts16_indoor_dfv4_path, 'w') as f:
        json.dump(C.seq_id_to_start_end_ts16_dfv4, f)
        print(C.start_end_ts16_indoor_dfv4_path, 'saved!')


# ------------------
#  Process all seqs
# ------------------
# ---------
#  Outdoor
# ---------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    process_save_sync_dfv4()
save_start_end_ts16_indoor_dfv4()
