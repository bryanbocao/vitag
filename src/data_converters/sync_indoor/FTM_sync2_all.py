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

        # -----
        #  FTM
        # -----
        self.FTM_path = self.seq_path + '/WiFi'
        self.FTM_dfv4_path = self.seq_path + '/FTM_dfv4' # ts13_dfv4 with offsets (os)
        if not os.path.exists(self.FTM_dfv4_path):
            os.makedirs(self.FTM_dfv4_path)
        self.subj_id_to_FTM_ts13_dfv4 = defaultdict()
        self.subj_id_to_FTM_ts13_dfv4_path = ''

        # -------------------
        #  Main data to save
        # -------------------
        self.subj_id_to_FTM_ts16_dfv4_path = ''
        self.subj_id_to_FTM_ts16_dfv4 = defaultdict()
        #  To de updated in update_parameters()
        # --------------------------------------


C = Config()

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(ls, K):
    return ls[min(range(len(ls)), key = lambda i: abs(float(ls[i])-float(K)))]

def update_parameters(seq_id_idx):
    C.seq_id = C.seq_id_ls[seq_id_idx]
    C.seq_path = C.seq_id_path_ls[seq_id_idx]
    C.img_type = 'RGBh_ts16_dfv4_anonymized'
    C.img_path = C.seq_path + '/' + C.img_type

    C.subjects = [15, 46, 77, 70, 73]
    C.subj_to_rand_id_file_path = C.exp_path + '/subj_to_rand_id.json'
    with open(C.subj_to_rand_id_file_path, 'r') as f:
        C.subj_to_id_dict = json.load(f)
        print(C.subj_to_rand_id_file_path, 'loaded!')
    C.seq_date = C.seq_id[:8]
    if '202012' in C.seq_date:
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

        # ------------------------------
        #  Load C.subj_id_to_FTM_ts13_dfv4
        # ------------------------------
        C.subj_id_to_FTM_ts13_dfv4_path = C.FTM_dfv4_path + '/subj_id_to_FTM_ts13_dfv4.json'
        with open(C.subj_id_to_FTM_ts13_dfv4_path, 'r') as f:
            C.subj_id_to_FTM_ts13_dfv4 = json.load(f)
            print(C.subj_id_to_FTM_ts13_dfv4_path + 'load!')

        # -------------------
        #  Main data to save
        # -------------------
        C.subj_id_to_FTM_ts16_dfv4_path = C.FTM_dfv4_path + '/subj_id_to_FTM_ts16_dfv4.json'
        C.subj_id_to_FTM_ts16_dfv4 = defaultdict()

def get_RGBh_ts16_dfv4_ls():
    # print(); print() # debug
    # print('C.img_path: ', C.img_path)
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
    # print();print() # debug
    # print('C.img_file_paths: ', C.img_file_paths)
    # print('C.img_names_ls: ', C.img_names_ls)
    # print('C.RGBh_ts16_dfv4_ls: ', C.RGBh_ts16_dfv4_ls)
    # e.g. 1608750783.029950

def convert_to_subj_id_to_FTM_ts16_dfv4_save():
    # ---------------------------
    #  Iterate Over All Subjects
    # ---------------------------
    for subj_i, subj in enumerate(C.subjects):
        subj_id = str(C.subj_to_id_dict['Indoor'][subj])
        # if subj == 'Others': continue
        C.subj_id_to_FTM_ts16_dfv4[subj_id] = defaultdict()

        ts13_dfv4_to_data = C.subj_id_to_FTM_ts13_dfv4[subj_id]
        FTM_ts13_dfv4_ls = list(ts13_dfv4_to_data.keys()); sorted(FTM_ts13_dfv4_ls)
        # print(); print() # debug
        # print('FTM_ts13_dfv4_ls: ', FTM_ts13_dfv4_ls)

        # --------------------------------
        #  Iterate Over All RGB_ts16_dfv4
        # --------------------------------
        for RGB_ts16_dfv4_i, RGB_ts16_dfv4 in enumerate(C.RGBh_ts16_dfv4_ls):
            C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4] = [[]]
            # if data_type_i == 0:
            #     C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4] = []

            # print();print() # debug
            # print('RGB_ts16_dfv4: ', RGB_ts16_dfv4) # e.g. 1608750783.362780

            # print(); print() # debug
            # print(ts13_dfv4_to_data.keys())
            FTM_ts13_dfv4 = closest(FTM_ts13_dfv4_ls, RGB_ts16_dfv4)
            # if FTM_ts13_dfv4 > ots26_to_ts16_dfv4(C.start_ots):
            #     print(); print() # debug
            #     print('FTM_ts13_dfv4: ', FTM_ts13_dfv4, ', RGB_ts16_dfv4: ', RGB_ts16_dfv4)
            '''
            e.g.
            FTM_ts13_dfv4:  1633372154.710 , RGB_ts16_dfv4:  1633372154.631903
            FTM_ts13_dfv4:  1633372154.710 , RGB_ts16_dfv4:  1633372154.731707
            FTM_ts13_dfv4:  1633372154.932 , RGB_ts16_dfv4:  1633372154.831894
            '''
            data = ts13_dfv4_to_data[FTM_ts13_dfv4]
            C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0].extend(data)

            print(); print() # debug
            print('seq_id:', C.seq_id)
            print('np.shape(C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0]):', \
                    np.shape(C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0]))
            print('C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0]:', \
                C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0])
            '''
            e.g.
            np.shape(C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0]): (2,)
            C.subj_id_to_FTM_ts16_dfv4[subj_id][RGB_ts16_dfv4][0]: [3470.0, 120.0]
            '''


    # ------
    #  Save
    # ------
    C.subj_id_to_FTM_ts16_dfv4_path = C.FTM_dfv4_path + '/subj_id_to_FTM_ts16_dfv4.json'
    with open(C.subj_id_to_FTM_ts16_dfv4_path, 'w') as f:
        json.dump(C.subj_id_to_FTM_ts16_dfv4, f)
        print(C.subj_id_to_FTM_ts16_dfv4_path + ' saved!')


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
        convert_to_subj_id_to_FTM_ts16_dfv4_save()
