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
        self.IMUlgyqm13_data_types = ['LINEAR', 'GYRO', 'Quaternion', 'MAG']
        self.IMUaccgym_data_types = ['ACCEL', 'GYRO', 'MAG']
        self.IMU_path = self.seq_path + '/IMU'
        self.IMU_dfv4_path = self.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.IMU_dfv4_path):
            os.makedirs(self.IMU_dfv4_path)
        self.subj_id_to_IMU19T_ts13_dfv4 = defaultdict()
        self.subj_id_to_IMU19T_ts13_dfv4_path = ''

        # -------------------
        #  Main data to save
        # -------------------
        self.subj_id_to_IMU19_ts16_dfv4_path = ''
        self.subj_id_to_IMU19_ts16_dfv4 = defaultdict()
        self.subj_id_to_IMUlgyq10_ts16_dfv4_path = ''
        self.subj_id_to_IMUlgyq10_ts16_dfv4 = defaultdict()
        self.subj_id_to_IMUlgyqm13_ts16_dfv4_path = ''
        self.subj_id_to_IMUlgyqm13_ts16_dfv4 = defaultdict()

        if self.IMU_200:
            self.subj_id_to_IMU19_200_ts16_dfv4 = defaultdict()
            self.subj_id_to_IMUaccgym_200_ts16_dfv4 = defaultdict()
        #  To de updated in update_parameters()
        # --------------------------------------


C = Config()

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(ls, K):
    return ls[min(range(len(ls)), key = lambda i: abs(float(ls[i])-float(K)))]

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

        C.img_type = 'RGBh_ts16_dfv4_anonymized'
        C.img_path = C.seq_path + '/' + C.img_type

        # -------
        #  IMU19
        # -------
        C.IMU_path = C.seq_path + '/IMU'
        C.IMU_dfv4_path = C.seq_path + '/IMU_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(C.IMU_dfv4_path):
            os.makedirs(C.IMU_dfv4_path)

        # ---------------------------------
        #  Load C.subj_id_to_IMU19T_ts13_dfv4
        # ---------------------------------
        C.subj_id_to_IMU19T_ts13_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19T_ts13_dfv4.json'
        with open(C.subj_id_to_IMU19T_ts13_dfv4_path, 'r') as f:
            C.subj_id_to_IMU19T_ts13_dfv4 = json.load(f)
            print(C.subj_id_to_IMU19T_ts13_dfv4_path + 'load!')

        # -------------------
        #  Main data to save
        # -------------------
        C.subj_id_to_IMU19_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_ts16_dfv4.json'
        C.subj_id_to_IMU19_ts16_dfv4 = defaultdict()
        C.subj_id_to_IMUlgyq10_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyq10_ts16_dfv4.json'
        C.subj_id_to_IMUlgyq10_ts16_dfv4 = defaultdict()
        C.subj_id_to_IMUlgyqm13_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyqm13_ts16_dfv4.json'
        C.subj_id_to_IMUlgyqm13_ts16_dfv4 = defaultdict()

        if C.IMU_200:
            C.subj_id_to_IMU19_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_200_ts16_dfv4.json'
            C.subj_id_to_IMU19_200_ts16_dfv4 = defaultdict()
            C.subj_id_to_IMUaccgym_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUaccgym_200_ts16_dfv4.json'
            C.subj_id_to_IMUaccgym_200_ts16_dfv4 = defaultdict()

def get_RGBh_ts16_dfv4_ls():
    # print(); print() # debug
    print('C.img_path: ', C.img_path)
    file_dir = pathlib.Path(C.img_path)
    file_pattern = "*.jpg"

    C.img_names_ls = []
    C.img_file_paths = []
    C.RGBh_ts16_dfv4_ls = []
    for file in file_dir.glob(file_pattern):
        # img_names_to_file_path[file.name] = file
        C.img_file_paths.append(file.__str__())
        # C.img_names_ls.append(file.name)
        # print(); print() # debug
        # print('file.name[:17]: ', file.name[:17]) # e.g. 1608750762.361545
        # C.RGBh_ts16_dfv4_ls.append(ots26_to_ts16_dfv4(file.name[:17]))
        C.RGBh_ts16_dfv4_ls.append(file.name[:17])
    C.img_file_paths.sort()
    # C.img_names_ls.sort()
    C.RGBh_ts16_dfv4_ls.sort()
    print();print() # debug
    print('C.img_file_paths: ', C.img_file_paths)
    print('C.img_names_ls: ', C.img_names_ls)
    print('C.RGBh_ts16_dfv4_ls: ', C.RGBh_ts16_dfv4_ls)
    # e.g. 1608750783.029950

def convert_to_subj_id_to_IMU_ts16_dfv4_save():
    # ---------------------------
    #  Iterate Over All Subjects
    # ---------------------------
    for subj_i, subj in enumerate(C.subjects):
        subj_id = str(C.subj_to_id_dict['Indoor'][subj])
        # if subj == 'Others': continue
        C.subj_id_to_IMU19_ts16_dfv4[subj_id] = defaultdict()
        C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id] = defaultdict()
        C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id] = defaultdict()
        if C.IMU_200:
            C.subj_id_to_IMU19_200_ts16_dfv4[subj_id] = defaultdict()
            C.subj_id_to_IMUaccgym_200_ts16_dfv4[subj_id] = defaultdict()
        # ----------------------------
        #  Iterate Over All data_type
        # ----------------------------
        for data_type_i, data_type in enumerate(C.IMU19_data_types):
            # print();print() # debug
            # print('C.subj_id_to_IMU19T_ts13_dfv4.keys(): ', C.subj_id_to_IMU19T_ts13_dfv4.keys())
            ts13_dfv4_to_data = C.subj_id_to_IMU19T_ts13_dfv4[subj_id][data_type]
            # print('data_type: ', data_type, ', ts13_dfv4: ', ts13_dfv4)
            IMU_ts13_dfv4_ls = list(ts13_dfv4_to_data.keys()); sorted(IMU_ts13_dfv4_ls)
            print(); print() # debug
            print('IMU_ts13_dfv4_ls: ', IMU_ts13_dfv4_ls)
            print('len(IMU_ts13_dfv4_ls): ', len(IMU_ts13_dfv4_ls)) # e.g. 11359
            print('len(C.RGBh_ts16_dfv4_ls): ', len(C.RGBh_ts16_dfv4_ls)) # e.g. 554

            # --------------------------------
            #  Iterate Over All RGBh_ts16_dfv4
            # --------------------------------
            for RGBh_ts16_dfv4 in C.RGBh_ts16_dfv4_ls:
                if data_type_i == 0:
                    C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4] = [[]]
                    C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4] = [[]]
                    C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4] = [[]]
                    if C.IMU_200:
                        C.subj_id_to_IMU19_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4] = [[] for _ in range(200)]
                        C.subj_id_to_IMUaccgym_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4] = [[] for _ in range(200)]

                # print();print() # debug
                # print('RGBh_ts16_dfv4: ', RGBh_ts16_dfv4) # e.g. 1608750783.362780

                # print(); print() # debug
                # print(ts13_dfv4_to_data.keys())
                IMU_ts13_dfv4 = closest(IMU_ts13_dfv4_ls, RGBh_ts16_dfv4)
                print(); print() # debug
                print('IMU_ts13_dfv4: ', IMU_ts13_dfv4, ', RGBh_ts16_dfv4: ', RGBh_ts16_dfv4)
                '''
                e.g.
                IMU_ts13_dfv4:  1608750656.696 , RGBh_ts16_dfv4:  1608750656.689327
                IMU_ts13_dfv4:  1608750657.023 , RGBh_ts16_dfv4:  1608750657.022574
                IMU_ts13_dfv4:  1608750657.364 , RGBh_ts16_dfv4:  1608750657.356593
                '''

                data = ts13_dfv4_to_data[IMU_ts13_dfv4]
                # print(); print() # debug
                # print('data_type: ', data_type, ', data: ', data)
                '''
                e.g.
                data_type:  ACCEL , data:  [-1.3266702, 3.6858606, 4.921017]
                data_type:  ACCEL , data:  [-2.8297122, 6.0826945, 11.293134]
                data_type:  ACCEL , data:  [-0.3337986, 4.3973093, 6.469884]
                '''
                C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4][0].extend(data)
                if data_type in C.IMUlgyq10_data_types:
                    C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4][0].extend(data)
                if data_type in C.IMUlgyqm13_data_types:
                    C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4][0].extend(data)

                if C.IMU_200:
                    # 200 data >>>
                    IMU_ts13_dfv4_idx = IMU_ts13_dfv4_ls.index(IMU_ts13_dfv4)
                    IMU_ts13_dfv4_200_ls = IMU_ts13_dfv4_ls[IMU_ts13_dfv4_idx : IMU_ts13_dfv4_idx + 200]
                    for i_, IMU_ts13_dfv4_ in enumerate(IMU_ts13_dfv4_200_ls):
                        data = ts13_dfv4_to_data[IMU_ts13_dfv4_]
                        C.subj_id_to_IMU19_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4][i_].extend(data)
                        if data_type in C.IMUaccgym_data_types:
                            C.subj_id_to_IMUaccgym_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4][i_].extend(data)
                    # 200 data <<<

                print(); print() # debug
                print('seq_id:', C.seq_id)
                print('subj_i:', subj_i, ', subj:', subj, ', data_type:', data_type)
                print('np.shape(C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4]):', \
                        np.shape(C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4]))
                print('C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4]:', \
                    C.subj_id_to_IMU19_ts16_dfv4[subj_id][RGBh_ts16_dfv4])
                print('np.shape(C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4]):', \
                        np.shape(C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4]))
                print('C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4]:', \
                    C.subj_id_to_IMUlgyq10_ts16_dfv4[subj_id][RGBh_ts16_dfv4])
                print('np.shape(C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4]):', \
                        np.shape(C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4]))
                print('C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4]:', \
                    C.subj_id_to_IMUlgyqm13_ts16_dfv4[subj_id][RGBh_ts16_dfv4])
                if C.IMU_200:
                    print('np.shape(C.subj_id_to_IMU19_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4]):', \
                        np.shape(C.subj_id_to_IMU19_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4]))
                    print('C.subj_id_to_IMUaccgym_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4]):', \
                        np.shape(C.subj_id_to_IMUaccgym_200_ts16_dfv4[subj_id][RGBh_ts16_dfv4]))


    # ------
    #  Save
    # ------
    C.subj_id_to_IMU19_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_ts16_dfv4.json'
    with open(C.subj_id_to_IMU19_ts16_dfv4_path, 'w') as f:
        json.dump(C.subj_id_to_IMU19_ts16_dfv4, f)
        print(C.subj_id_to_IMU19_ts16_dfv4_path + ' saved!')
    C.subj_id_to_IMUlgyq10_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyq10_ts16_dfv4.json'
    with open(C.subj_id_to_IMUlgyq10_ts16_dfv4_path, 'w') as f:
        json.dump(C.subj_id_to_IMUlgyq10_ts16_dfv4, f)
        print(C.subj_id_to_IMUlgyq10_ts16_dfv4_path + ' saved!')
    C.subj_id_to_IMUlgyqm13_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUlgyqm13_ts16_dfv4.json'
    with open(C.subj_id_to_IMUlgyqm13_ts16_dfv4_path, 'w') as f:
        json.dump(C.subj_id_to_IMUlgyqm13_ts16_dfv4, f)
        print(C.subj_id_to_IMUlgyqm13_ts16_dfv4_path + ' saved!')
    if C.IMU_200:
        C.subj_id_to_IMU19_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMU19_200_ts16_dfv4.json'
        with open(C.subj_id_to_IMU19_200_ts16_dfv4_path, 'w') as f:
            json.dump(C.subj_id_to_IMU19_200_ts16_dfv4, f)
            print(C.subj_id_to_IMU19_200_ts16_dfv4_path + ' saved!')
        C.subj_id_to_IMUaccgym_200_ts16_dfv4_path = C.IMU_dfv4_path + '/subj_id_to_IMUaccgym_200_ts16_dfv4.json'
        with open(C.subj_id_to_IMUaccgym_200_ts16_dfv4_path, 'w') as f:
            json.dump(C.subj_id_to_IMUaccgym_200_ts16_dfv4, f)
            print(C.subj_id_to_IMUaccgym_200_ts16_dfv4_path + ' saved!')

# ------------------
#  Process all seqs
# ------------------
# --------
#  indoor
# --------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    if '202012' in C.seq_date:
        get_RGBh_ts16_dfv4_ls()
        convert_to_subj_id_to_IMU_ts16_dfv4_save()
