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

        self.img_type = 'RGB_ts16_dfv4_anonymized'
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


        # -----
        #  GND
        # -----
        self.GND_path = self.seq_path + '/GND/RAN_' + self.seq_id + '-export.json'
        with open(self.GND_path, 'r') as f:
            self.GND_dict = json.load(f)
            # print();print() # debug
            # print('self.GND_dict: ', self.GND_dict)
            '''
            e.g.
            {'id': '8u1F_c4vs', 'type': 'RECTANGLE', 'tags': ['Murtadha'],
            'boundingBox': {'height': 147.33590733590734, 'width': 78.72648335745296,
            'left': 276.931982633864, 'top': 65.5019305019305}, 'points': [{'x':
            276.931982633864, 'y': 65.5019305019305}, {'x': 355.6584659913169,
            'y': 65.5019305019305}, {'x': 355.6584659913169, 'y': 212.83783783783784},
            {'x': 276.931982633864, 'y': 212.83783783783784}]}], 'version': '2.1.0'}}}
            '''

        self.GND_ts16_dfv4_path = self.seq_path + '/GND_ts16_dfv4'
        if not os.path.exists(self.GND_ts16_dfv4_path):
            os.makedirs(self.GND_ts16_dfv4_path)

        # -------------------
        #  Main data to save
        # -------------------
        self.GND_ts16_dfv4_to_BBX4tlhw_path = self.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX4tlhw.json'
        self.GND_ts16_dfv4_to_BBX4tlhw = defaultdict()
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

    C.img_path = C.seq_path + '/' + C.img_type

    # -----
    #  GND
    # -----
    C.GND_path = C.seq_path + '/GND/RAN_' + C.seq_id + '-export.json'
    with open(C.GND_path, 'r') as f:
        C.GND_dict = json.load(f)
        # print();print() # debug
        # print('C.GND_dict: ', C.GND_dict)
        '''
        e.g.
        {'id': '8u1F_c4vs', 'type': 'RECTANGLE', 'tags': ['Murtadha'],
        'boundingBox': {'height': 147.33590733590734, 'width': 78.72648335745296,
        'left': 276.931982633864, 'top': 65.5019305019305}, 'points': [{'x':
        276.931982633864, 'y': 65.5019305019305}, {'x': 355.6584659913169,
        'y': 65.5019305019305}, {'x': 355.6584659913169, 'y': 212.83783783783784},
        {'x': 276.931982633864, 'y': 212.83783783783784}]}], 'version': '2.1.0'}}}
        '''

    C.GND_ts16_dfv4_path = C.seq_path + '/GND_ts16_dfv4'
    if not os.path.exists(C.GND_ts16_dfv4_path):
        os.makedirs(C.GND_ts16_dfv4_path)

    # -------------------
    #  Main data to save
    # -------------------
    C.GND_ts16_dfv4_to_BBX4tlhw_path = C.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX4tlhw.json'
    C.GND_ts16_dfv4_to_BBX4tlhw = defaultdict()

def get_RGB_ts16_dfv4_ls():
    file_dir = pathlib.Path(C.img_path)
    file_pattern = "*.jpg"

    C.img_names_ls = []
    C.img_file_paths = []
    C.RGB_ts16_dfv4_ls = []
    for file in file_dir.glob(file_pattern):
        # img_names_to_file_path[file.name] = file
        C.img_file_paths.append(file.__str__())
        # C.img_names_ls.append(file.name)
        C.RGB_ts16_dfv4_ls.append(file.name[:17])
    C.img_file_paths.sort()
    # C.img_names_ls.sort()
    C.RGB_ts16_dfv4_ls.sort()
    # print();print() # debug
    # print('C.img_file_paths: ', C.img_file_paths)
    # print('C.img_names_ls: ', C.img_names_ls)
    # print('C.RGB_ts16_dfv4_ls: ', C.RGB_ts16_dfv4_ls)
    # e.g. 1608750783.029950

def convert_to_BBX4tlhw_save():
    # https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
    def closest(lst, K):
        return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-float(K)))]
    for k in C.GND_dict['assets'].keys():
        ots26 = C.GND_dict['assets'][k]['asset']['name']
        ots26 = ots26[:10] + ' ' + ots26[13:-4]
        # print();print() # debug
        # print('ots26: ', ots26) # e.g. 2021-10-04 14:31:27.338290

        if '_' in ots26:
            ots26 = ots26.replace('_', ':')
        ts16_dfv4 = ots26_to_ts16_dfv4(ots26)
        # print();print() # debug
        # print('ts16_dfv4: ', ts16_dfv4) # e.g. 1633372287.338290
        if ts16_dfv4 in C.RGB_ts16_dfv4_ls:
            for reg in C.GND_dict['assets'][k]['regions']:
                # print();print() # debug
                # print('reg: ', reg)
                subj = reg['tags'][0]
                # if subj == 'Others' or 'Unknown': subj = 'Other' + str(reg['id'])
                if subj not in C.subjects: subj = 'Other' + str(reg['id'])
                data = {'id': reg['id'], 't': reg['boundingBox']['top'], 'l': reg['boundingBox']['left'], \
                        'h': reg['boundingBox']['height'], 'w': reg['boundingBox']['width']}
                if ts16_dfv4 not in C.GND_ts16_dfv4_to_BBX4tlhw.keys():
                    C.GND_ts16_dfv4_to_BBX4tlhw[ts16_dfv4] = defaultdict()
                C.GND_ts16_dfv4_to_BBX4tlhw[ts16_dfv4][subj] = data

    # print();print() # debug
    # print('C.GND_ts16_dfv4_to_BBX4tlhw: ', C.GND_ts16_dfv4_to_BBX4tlhw)
    with open(C.GND_ts16_dfv4_to_BBX4tlhw_path, 'w') as f:
        json.dump(C.GND_ts16_dfv4_to_BBX4tlhw, f)
        print();print() # debug
        print('C.GND_ts16_dfv4_to_BBX4tlhw: ', C.GND_ts16_dfv4_to_BBX4tlhw)
        print(C.GND_ts16_dfv4_to_BBX4tlhw_path, 'saved!')



# ------------------
#  Process all seqs
# ------------------
# ---------
#  Outdoor
# ---------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    get_RGB_ts16_dfv4_ls()
    convert_to_BBX4tlhw_save()
