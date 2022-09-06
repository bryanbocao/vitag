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
#  Configurations of the whole experiment
# ----------------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        self.user = 'brcao' # 'anon'
        self.root_path = '/media/' + self.user + '/eData2' # '/home/' + self.user
        self.seq_root_path = self.root_path + '/Data/datasets/RAN/seqs/indoor/scene0'
        self.IMU_200 = False # edit
        if self.IMU_200: self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN4model_dfv4_IMU_200'
        else: self.dataset4model_root_path = self.root_path + '/Data/datasets/RAN4model_dfv4'
        self.seq4model_root_path = self.dataset4model_root_path + '/seqs/indoor/scene0'
        if not os.path.exists(self.seq4model_root_path):
            os.makedirs(self.seq4model_root_path)
        print('self.seq_root_path: ', self.seq_root_path)
        self.seq_id_path_ls = glob.glob(self.seq_root_path + '/*')
        self.seq_id_ls = [seq_id_path[-15:] for seq_id_path in self.seq_id_path_ls]

        # --------------------------------------
        #  To be updated in update_parameters()
        self.seq_path = self.seq_id_path_ls[0]
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)
        self.exp = 1 # edit
        self.exp_path = self.root_path + '/Data/datasets/RAN/seqs/exps/exp' + str(self.exp)

        self.subjects = [15, 46, 77, 70, 73]
        self.subj_to_rand_id_file_path = self.exp_path + '/subj_to_rand_id.json'
        with open(self.subj_to_rand_id_file_path, 'r') as f:
            self.subj_to_id_dict = json.load(f)
            print(self.subj_to_rand_id_file_path, 'loaded!')
        self.phone_time_offsets = [0] * len(self.subjects)

        print(); print() # debug
        print('self.subjects: ', self.subjects)

        # -------
        #  IMU19
        # -------
        self.IMU19_data_types = ['ACCEL', 'GYRO', 'MAG', 'GRAV', 'LINEAR', 'Quaternion'] # ['ACCEL', 'GRAV', 'LINEAR', 'Quaternion', 'MAG', 'GYRO']
        self.IMUlgyq10_data_types = ['LINEAR', 'GYRO', 'Quaternion']
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv4_path = self.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.IMU_dfv4_path):
            os.makedirs(self.IMU_dfv4_path)
        self.subj_id_to_IMU19T_ts13_dfv4 = defaultdict()
        #  To de updated in update_parameters()
        # --------------------------------------

C = Config()

def update_parameters(seq_id_idx):
    C.seq_id = C.seq_id_ls[seq_id_idx]
    C.seq_path = C.seq_id_path_ls[seq_id_idx]
    C.subjects = [15, 46, 77, 70, 73]
    C.seq_date = C.seq_id[:8]
    if '202012' in C.seq_date:
        if C.seq_date == '20201228':
            C.phone_time_offsets = [2.219273, 2.962273, 2.764273, 3.461273, 2.948273]
        print(); print() # debug
        print('C.subjects: ', C.subjects)
        print('C.seq_date: ', C.seq_date)
        print('C.phone_time_offsets: ', C.phone_time_offsets)

        # -------
        #  IMU19
        # -------
        C.IMU_path = C.seq_path + '/IMU'
        C.IMU_dfv4_path = C.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(C.IMU_dfv4_path):
            os.makedirs(C.IMU_dfv4_path)

def convert_to_IMU19T_to_ts13_dfv4_save():
    IMU_files_paths = glob.glob(C.IMU_path + '/*')
    print(); print() # debug
    print(glob.glob(C.IMU_path + '/*'))
    # ---------------------------
    #  Iterate Over All Subjects
    # ---------------------------
    for subj_i, subj in enumerate(C.subjects):
        subj_id = str(C.subj_to_id_dict['Indoor'][subj])
        # --------------------------------
        #  Init C.subj_id_to_IMU19T_ts13_dfv4
        # --------------------------------
        C.subj_id_to_IMU19T_ts13_dfv4[subj_id] = defaultdict() # 'T' stands for Data Type
        for data_type in C.IMU19_data_types:
            C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type] = defaultdict()

        # ------
        #  Load
        # ------
        print(); print() # debug
        print('IMU_files_paths: ', IMU_files_paths, ', subj: ', subj)
        C.IMU_subj_data_path = [IMU_file_path for IMU_file_path in IMU_files_paths if subj in IMU_file_path][0]
        print(); print() # debug
        print('C.IMU_subj_data_path: ', C.IMU_subj_data_path)

        with open(C.IMU_subj_data_path, 'r') as f:
            csv_reader = reader(f)
            for i, row in enumerate(csv_reader):
                if i > 0 and row[1] in C.IMU19_data_types:
                    # print('row: ', row)
                    # e.g. row:  ['1608750777299', 'ACCEL', '-0.8912008', '2.24727', '8.2579565']
                    ts13_dfv4 = ot_ts = row[0]
                    # print();print() # debug
                    # print('ts13_dfv4: ', ts13_dfv4)
                    # e.g. 1609192767714 # 13 digits
                    # 10 digits in seconds

                    # with offsets
                    ts13_dfv4 = str(int(ot_ts[:10]) + int(C.phone_time_offsets[subj_i])) + '.' + ot_ts[10:]
                    # print();print() # debug
                    # print('ts13_dfv4 applied with offsets: ', ts13_dfv4) # e.g. 1608750727.217

                    data_type = row[1]
                    # print();print() # debug
                    # print('data_type: ', data_type)

                    data = []
                    for data_ in row[2:]:
                        data.append(float(data_))
                    # print();print() # debug
                    # print('data: ', data)

                    C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type][ts13_dfv4] = data

        # ------------------------------------------------------------
        #  Verify data, each number should be approximately the same.
        # ------------------------------------------------------------
        print();print() # debug
        print('subj:', subj)
        # print('C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type]: ', C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type])
        for data_type in C.IMU19_data_types:
            print(data_type, len(C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type]))

    # ------
    #  Save
    # ------
    C.subj_id_to_IMU19T_ts13_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19T_ts13_dfv4.json'
    with open(C.subj_id_to_IMU19T_ts13_dfv4_path, 'w') as f:
        json.dump(C.subj_id_to_IMU19T_ts13_dfv4, f)
        print(C.subj_id_to_IMU19T_ts13_dfv4_path + ' saved!')


# ------------------
#  Process all seqs
# ------------------
# --------
#  indoor
# --------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    if '202012' in C.seq_date:
        convert_to_IMU19T_to_ts13_dfv4_save()
