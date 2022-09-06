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

        self.GND_ts16_dfv4_to_BBX4tlhw_path = self.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX4tlhw.json'
        self.GND_ts16_dfv4_to_BBX4tlhw = defaultdict()

        # -------------------
        #  Main data to save
        # -------------------
        self.RGB_ts16_dfv4_ls_path = self.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
        self.RGB_ts16_dfv4_ls = []
        self.GND_ts16_dfv4_to_BBX5_dfv4_path = self.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX5_dfv4.json'
        self.GND_ts16_dfv4_to_BBX5_dfv4 = defaultdict()
        #  To de updated in update_parameters()
        # --------------------------------------

C = Config()

# --------------
#  Access Depth
# --------------
def get_depth_img_names_ls_save_RGB_ts16_dfv4_ls():
    file_dir = pathlib.Path(C.img_path)
    file_pattern = "*.pkl"

    C.depth_img_names_ls = []
    C.img_file_paths = []
    C.RGB_ts16_dfv4_ls = []
    for file in file_dir.glob(file_pattern):
        C.img_file_paths.append(file.__str__())
        C.depth_img_names_ls.append(file.name)
        C.RGB_ts16_dfv4_ls.append(ots26_to_ts16_dfv4(file.name[:-4]))
    C.img_file_paths.sort()
    C.depth_img_names_ls.sort()
    C.RGB_ts16_dfv4_ls.sort()
    print(); print() # debug
    print('C.img_file_paths: ', C.img_file_paths)
    print('C.depth_img_names_ls: ', C.depth_img_names_ls)
    '''
    e.g.
    '2020-12-23 14:27:20.476423.pkl', '2020-12-23 14:27:20.809450.pkl', '2020-12-23 14:27:21.142523.pkl'
    '''
    # print('C.RGB_ts16_dfv4_ls: ', C.RGB_ts16_dfv4_ls)
    with open(C.RGB_ts16_dfv4_ls_path, 'w') as f:
        json.dump(C.RGB_ts16_dfv4_ls, f, cls=NpEncoder)
        print(C.RGB_ts16_dfv4_ls_path, 'saved!')

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

    # -----
    #  GND
    # -----
    C.GND_ts16_dfv4_path = C.seq_path + '/GND_ts16_dfv4'
    if not os.path.exists(C.GND_ts16_dfv4_path): os.makedirs(C.GND_ts16_dfv4_path)

    C.GND_ts16_dfv4_to_BBX4tlhw_path = C.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX4tlhw.json'
    with open(C.GND_ts16_dfv4_to_BBX4tlhw_path, 'r') as f:
        C.GND_ts16_dfv4_to_BBX4tlhw = json.load(f)
        print(C.GND_ts16_dfv4_to_BBX4tlhw_path, 'loaded!')

    # -------------------
    #  Main data to save
    # -------------------
    C.RGB_ts16_dfv4_ls_path = C.seq_path_for_model + '/RGB_ts16_dfv4_ls.json'
    C.RGB_ts16_dfv4_ls = []
    C.GND_ts16_dfv4_to_BBX5_dfv4_path = C.GND_ts16_dfv4_path + '/GND_ts16_dfv4_to_BBX5_dfv4.json'
    C.GND_ts16_dfv4_to_BBX5_dfv4 = defaultdict()


def convert_to_GND_ts16_dfv4_to_BBX5_dfv4_save():
    for i, img_name in enumerate(C.depth_img_names_ls):
        depth = pickle.load(open(C.img_file_paths[i], "rb"))
        # print('np.shape(depth): ', np.shape(depth)) # e.g. (720, 1280)
        # print();print() # debug
        # print('img_name: ', img_name) # e.g. 2020-12-23 14:27:20.142277.pkl
        ots26 = img_name[:-4]
        # print();print() # debug
        # print('ots26: ', ots26) # e.g. 2020-12-23 14:27:19.809528
        ts16_dfv4 = ots26_to_ts16_dfv4(ots26)
        # print();print() # debug
        # print('ts16_dfv4: ', ts16_dfv4) # e.g. 1609191423.242913

        if ts16_dfv4 >= ots26_to_ts16_dfv4(C.start_ots) and ts16_dfv4 < ots26_to_ts16_dfv4(C.end_ots) \
            and ts16_dfv4 in C.GND_ts16_dfv4_to_BBX4tlhw:
            # ----------
            #  Load img
            # ----------
            RGB_img_path = C.seq_path + '/RGB_ts16_dfv4_anonymized/' + ts16_dfv4 + '_anonymized.jpg'
            print();print() # debug
            print('RGB_img_path: ', RGB_img_path)
            print('i: ', i, ', ts16_dfv4: ', ts16_dfv4)
            img = cv2.imread(RGB_img_path)
            # pil_im = Image.open(RGB_img_path)
            # pil_im.show()

            BBX4tlhw = C.GND_ts16_dfv4_to_BBX4tlhw[ts16_dfv4]
            # print();print() # debug
            # print('BBX4tlhw: ', BBX4tlhw)

            C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4] = defaultdict()
            for subj in BBX4tlhw:
                GND_data = BBX4tlhw[subj]
                id, l, t, w, h = GND_data['id'], int(GND_data['l']), int(GND_data['t']), int(GND_data['w']), int(GND_data['h'])
                # print(); print() # debug
                # print('id: ', id)

                subj_i = -1 if 'Other' in subj else C.subjects.index(subj)

                c_row = t + int(h / 2) # float(l) + (float(w) / 2)
                c_col = l + int(w / 2) # float(t) + (float(h) / 2)

                d = depth[c_row, c_col] # Note (row, col) or (y, x) here
                # print();print() # debug
                # print('d: ', d, ', c_row: ,', c_row, ', c_col: ', c_col) # should not have nan

                # -------------------------------------------------------
                #  Verify reg
                #  Comment this block out when saving without displaying
                # -------------------------------------------------------
                subj_color = C.color_dict[C.color_ls[subj_i]]
                img = cv2.circle(img, (c_col, c_row), 10, subj_color, 10) # Note (col, row) or (x, y) here
                img = cv2.putText(img, str(subj) + '(' + str(c_col) + ',' + str(c_row) + \
                                    ',' + str(d)[:4] + ')', (c_col + 25, c_row), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, subj_color, 2, cv2.LINE_AA)
                img = cv2.rectangle(img, (int(l), int(t)), ((int(l + w), int(t + h))), subj_color, 2)
                C.GND_ts16_dfv4_to_BBX5_dfv4[ts16_dfv4][subj] = {'subj': subj, 'c_row': c_row, 'c_col': c_col, 'w': w, 'h': h, 'd': d}

            # ---------
            #  Display
            # ---------
            # cv2.imshow("img", img); cv2.waitKey(0) # edit

    with open(C.GND_ts16_dfv4_to_BBX5_dfv4_path, 'w') as f:
        json.dump(C.GND_ts16_dfv4_to_BBX5_dfv4, f, cls=NpEncoder)
        print(C.GND_ts16_dfv4_to_BBX5_dfv4_path, 'saved!')

# ------------------
#  Process all seqs
# ------------------
# ---------
#  Outdoor
# ---------
for seq_id_idx, seq_id in enumerate(C.seq_id_ls):
    update_parameters(seq_id_idx)
    get_depth_img_names_ls_save_RGB_ts16_dfv4_ls()
    convert_to_GND_ts16_dfv4_to_BBX5_dfv4_save()
