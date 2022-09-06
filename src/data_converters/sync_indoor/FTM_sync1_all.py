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
        #  FTM
        # -------
        '''
        0: time, 1: 'WiFi-FTM', 2: MAC address, 3: rangingResult.getDistanceMm(),
        4: rangingResult.getDistanceStdDevMm(), 5: rangingResult.getNumAttemptedMeasurements(),
        6: rangingResult.getNumSuccessfulMeasurements(),
        7: rangingResult.getRangingTimestampMillis(), 8: rangingResult.getRssi(),
        9: rangingResult.getStatus(), 10: rangingResult.hashCode()
        The definition of the returned values can be found https://developer.android.com/reference/android/net/wifi/rtt/RangingResult
        '''
        self.FTM_path = self.seq_path + '/WiFi'
        self.FTM_dfv4_path = self.seq_path + '/FTM_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.FTM_dfv4_path):
            os.makedirs(self.FTM_dfv4_path)
        self.subj_id_to_FTM_ts13_dfv4 = defaultdict()
        #  To de updated in update_parameters()
        # --------------------------------------

C = Config()

def update_parameters(seq_id_idx):
    C.seq_id = C.seq_id_ls[seq_id_idx]
    C.seq_path = C.seq_id_path_ls[seq_id_idx]
    C.subjects = [15, 46, 77, 70, 73]
    C.subj_to_rand_id_file_path = C.exp_path + '/subj_to_rand_id.json'
    with open(C.subj_to_rand_id_file_path, 'r') as f:
        C.subj_to_id_dict = json.load(f)
        print(C.subj_to_rand_id_file_path, 'loaded!')
    C.seq_date = C.seq_id[:8]
    if C.seq_date == '20201228':
        C.phone_time_offsets = [2.219273, 2.962273, 2.764273, 3.461273, 2.948273]
    print(); print() # debug
    print('C.subjects: ', C.subjects)
    print('C.seq_date: ', C.seq_date)
    print('C.phone_time_offsets: ', C.phone_time_offsets)

    # -----
    #  FTM
    # -----
    C.FTM_path = C.seq_path + '/WiFi'
    C.FTM_dfv4_path = C.seq_path + '/FTM_dfv4' # ts13_dfv4 with offsets (os)
    if not os.path.exists(C.FTM_dfv4_path):
        os.makedirs(C.FTM_dfv4_path)
    C.subj_id_to_FTM_ts13_dfv4 = defaultdict()
    # C.start_ots, C.end_ots, C.subj_to_offset = get_start_end_ts_update_phone_with_offsets(C.seq_id, C.meta_path)


def convert_to_FTM_to_ts13_dfv4_save():
    FTM_files_paths = glob.glob(C.FTM_path + '/*')
    print(); print() # debug
    verify_paths = glob.glob(C.FTM_path + '/*')
    print('glob.glob(C.FTM_path + /*): ', verify_paths)
    if len(verify_paths) > 0:
        # ---------------------------
        #  Iterate Over All Subjects
        # ---------------------------
        for subj_i, subj in enumerate(C.subjects):
            subj_id = str(C.subj_to_id_dict['Indoor'][subj])
            # --------------------------------
            #  Init C.subj_id_to_FTM_ts13_dfv4
            # --------------------------------
            C.subj_id_to_FTM_ts13_dfv4[subj_id] = defaultdict() # 'T' stands for Data Type

            # ------
            #  Load
            # ------
            # print(); print() # debug
            # print('FTM_files_paths: ', FTM_files_paths, ', subj: ', subj)
            C.FTM_subj_data_path = [FTM_file_path for FTM_file_path in FTM_files_paths if subj in FTM_file_path][0]
            # print(); print() # debug
            # print('C.FTM_subj_data_path: ', C.FTM_subj_data_path)

            with open(C.FTM_subj_data_path, 'r') as f:
                csv_reader = reader(f)
                for i, row in enumerate(csv_reader):
                    # print('row: ', row)
                    # e.g. row:  ['1633372095808', 'WiFi-FTM', ' 28:bd:89:0e:7d:fa',
                    #            '304.0', '609.0', '8.0', '7.0', '5.3334497E9', '-45.0', '0.0', '-1.89970384E8']
                    ts13_dfv4 = ot_ts = row[0]
                    # print();print() # debug
                    # print('ts13_dfv4: ', ts13_dfv4)
                    # e.g. 1609192767714 # 13 digits
                    # 10 digits in seconds

                    # with offsets
                    ts13_dfv4 = str(int(ot_ts[:10]) + int(C.phone_time_offsets[subj_i])) + '.' + ot_ts[10:]
                    # print();print() # debug
                    # print('ts13_dfv4 applied with offsets: ', ts13_dfv4) # e.g. 1608750727.217

                    data = [float(row[3]), float(row[4])]
                    # print();print() # debug
                    # print('data: ', data)

                    C.subj_id_to_FTM_ts13_dfv4[subj_id][ts13_dfv4] = data

            # ------------------------------------------------------------
            #  Verify data, each number should be approximately the same.
            # ------------------------------------------------------------
            print();print() # debug
            print('subj:', subj)
            # print('C.subj_id_to_FTM_ts13_dfv4[subj][data_type]: ', C.subj_id_to_FTM_ts13_dfv4[subj][data_type])

        # ------
        #  Save
        # ------
        C.subj_id_to_FTM_ts13_dfv4_path = C.FTM_dfv4_path + '/subj_id_to_FTM_ts13_dfv4.json'
        with open(C.subj_id_to_FTM_ts13_dfv4_path, 'w') as f:
            json.dump(C.subj_id_to_FTM_ts13_dfv4, f)
            print(C.subj_id_to_FTM_ts13_dfv4_path + ' saved!')


# ------------------
#  Process all seqs
# ------------------
# --------
#  indoor
# --------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    convert_to_FTM_to_ts13_dfv4_save()
