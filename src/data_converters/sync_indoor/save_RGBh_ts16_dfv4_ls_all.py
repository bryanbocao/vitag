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
        self.seq_id = self.seq_id_ls[0]
        self.seq_path_for_model = self.seq4model_root_path + '/' + self.seq_id
        print(); print() # debug
        print('self.seq_id_path_ls: ', self.seq_id_path_ls)
        print('self,seq_id_ls: ', self.seq_id_ls)
        self.exp = 1 # edit
        self.exp_path = self.root_path + '/Data/datasets/RAN/seqs/exps/exp' + str(self.exp)
        self.meta_path = self.exp_path + '/meta_indoor.csv'
        self.seq_id_to_start_end_ts16_dfv4 = defaultdict()
        with open(self.meta_path, 'r') as f:
            csv_reader = reader(f)
            for i, row in enumerate(csv_reader):
                if i > 0:
                    print('row: ', row)
                    self.seq_id_to_start_end_ts16_dfv4[row[0]] = defaultdict()
                    self.seq_id_to_start_end_ts16_dfv4[row[0]]['start'] = row[1]
                    self.seq_id_to_start_end_ts16_dfv4[row[0]]['end'] = row[2]
        self.start_end_ts16_indoor_dfv4_folder_path = self.dataset4model_root_path + \
            '/exps/exp' + str(self.exp)
        if not os.path.exists(self.start_end_ts16_indoor_dfv4_folder_path):
            os.makedirs(self.start_end_ts16_indoor_dfv4_folder_path)
        self.start_end_ts16_indoor_dfv4_path = self.start_end_ts16_indoor_dfv4_folder_path + '/start_end_ts16_dfv4_indoor.json'

        # print('\n\n')
        # print('self.seq_id_to_start_ts16: ', self.seq_id_to_start_ts16)
        # print('self.seq_id_to_end_ts16: ', self.seq_id_to_end_ts16)

        # -------------------
        #  Main data to save
        # -------------------
        self.RGBh_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
        self.RGB_ts16_dfv4_ls = []
        #  To de updated in update_parameters()
        # --------------------------------------


C = Config()

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(ls, K):
    return ls[min(range(len(ls)), key = lambda i: abs(float(ls[i])-float(K)))]

def update_parameters(seq_id_idx):
    C.seq_id = C.seq_id_ls[seq_id_idx]
    C.seq_path = C.seq_id_path_ls[seq_id_idx]
    print(); print() # debug

    C.img_type = 'RGBh_ts16_dfv4_anonymized'
    C.img_path = C.seq_path + '/' + C.img_type
    C.seq_path_for_model = C.seq4model_root_path + '/' + C.seq_id
    if not os.path.exists(C.seq_path_for_model):
        os.makedirs(C.seq_path_for_model)

    # -------------------
    #  Main data to save
    # -------------------
    C.RGBh_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGBh_ts16_dfv4_ls.json'
    C.RGB_ts16_dfv4_ls = []

def get_save_RGBh_ts16_dfv4_ls():
    # print(); print() # debug
    print('C.img_path: ', C.img_path)
    file_dir = pathlib.Path(C.img_path)
    file_pattern = "*.jpg"

    C.img_names_ls = []
    C.img_file_paths = []
    C.RGBh_ts16_dfv4_ls = []
    for file in file_dir.glob(file_pattern):

        ts16_dfv4 = file.name[:17]
        if ts16_dfv4 >= C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['start'] and \
            ts16_dfv4 < C.seq_id_to_start_end_ts16_dfv4[C.seq_id]['end']:
            # img_names_to_file_path[file.name] = file
            C.img_file_paths.append(file.__str__())
            # C.img_names_ls.append(file.name)
            # print(); print() # debug
            # print('file.name[:17]: ', file.name[:17]) # e.g. 1608750762.361545
            # C.RGBh_ts16_dfv4_ls.append(ots26_to_ts16_dfv4(file.name[:17]))
            C.RGBh_ts16_dfv4_ls.append(ts16_dfv4) # file.name[:17])
    C.img_file_paths.sort()
    # C.img_names_ls.sort()
    C.RGBh_ts16_dfv4_ls.sort()
    # print();print() # debug
    # print('C.img_file_paths: ', C.img_file_paths)
    # print('C.img_names_ls: ', C.img_names_ls)
    # print('C.RGBh_ts16_dfv4_ls: ', C.RGBh_ts16_dfv4_ls)
    # e.g. 1608750783.029950
    with open(C.RGBh_ts16_dfv4_ls_path, 'w') as f:
        json.dump(C.RGBh_ts16_dfv4_ls, f)
        # print('C.RGBh_ts16_dfv4_ls: ', C.RGBh_ts16_dfv4_ls)
        print(C.RGBh_ts16_dfv4_ls_path, 'saved!')

def save_start_end_ts16_indoor_dfv4():
    with open(C.start_end_ts16_indoor_dfv4_path, 'w') as f:
        json.dump(C.seq_id_to_start_end_ts16_dfv4, f)
        print(C.start_end_ts16_indoor_dfv4_path, 'saved!')

# ------------------
#  Process all seqs
# ------------------
# --------
#  indoor
# --------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    if '202012' in C.img_path:
        get_save_RGBh_ts16_dfv4_ls()
    save_start_end_ts16_indoor_dfv4()
