# ViTag

Repository of our paper accepted in [SECON 2022](https://secon2022.ieee-secon.org/program/):

**Bryan Bo Cao**, Abrar Alali, Hansi Liu, Nicholas Meegan, Marco Gruteser, Kristin Dana, Ashwin Ashok, Shubham Jain, **ViTag: Online WiFi Fine Time Measurements Aided Vision-Motion Identity Association in Multi-person Environments**, 2022 19th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON).

**Bryan Bo Cao**, Abrar Alali, Hansi Liu, Nicholas Meegan, Marco Gruteser, Kristin Dana, Ashwin Ashok, Shubham Jain, **Demo: Tagging Vision with Smartphone Identities by Vision2Phone Translation**, 2022 19th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON).
Received **[Best Demonstration Award](https://secon2022.ieee-secon.org/program/)**

# Vi-Fi Dataset

**New** 01/16/2024: We released the synchronized version (**RAN4model_dfv4p4**) of our data for future usage. This version is convenient for your research without undergoing preprocessing the raw data again. Check out the details in the [DATA.md](https://github.com/bryanbocao/vitag/blob/main/DATA.md) file.

[Dataset(raw data) link](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)

[paperswithcode link](https://paperswithcode.com/dataset/vi-fi-multi-modal-dataset)

## Abstract
We demonstrate our system _ViTag_ to associate user identities across multimodal data from cameras and smartphones. _ViTag_ associates a sequence of vision tracker generated bounding boxes with Inertial Measurement Unit (IMU) data and Wi-Fi Fine Time Measurements (FTM) from smartphones. Our system first performs cross-modal translation using a multimodal LSTM encoder-decoder network (_X-Translator_) that translates one modality to another, e.g. reconstructing IMU and FTM readings purely from camera bounding boxes. Next, an association module finds identity matches between camera and phone domains, where the translated modality is then matched with the observed data from the same modality. Our system performs in real-world indoor and outdoor environments and achieves an average Identity Precision Accuracy (IDP) of 88.39% on a 1 to 3 seconds window. Further study on modalities within the phone domain shows the FTM can improve association performance by 12.56% on average.

## Motivation
Associating visually detected subjects with corresponding phone identifiers using multimodal data.
<img src="https://user-images.githubusercontent.com/14010288/182496142-ad216041-b6b4-427c-bc37-d9e40fb5b56b.jpg" width="420">

## System Overview
The system first translates data from camera domain to phone domain using the proposed model _X-Translator_, then it finds the matching between the reconstructed and observed phone data. Vision tracklets (_T<sub>c</sub>_) are fed into _X-Translator_ to reconstruct the corresponding phone tracklets for IMU (_T<sub>i</sub>'_) and FTM (_T<sub>f</sub>'_). We demonstrate _ViTag_’s association capacity by visualizing the estimated Vision-Phone correspondences in the video stream.
![system_overview](https://user-images.githubusercontent.com/14010288/182496181-bef70770-8aea-422b-845b-bfba84293240.jpg)

## X-Translator
_X-Translator_ architecture: A bidirectional LSTM based encoder-decoder model. Encoders are used to learn unimodal representations from vision tracklets (_T<sub>c</sub>_) and IMU data (_T<sub>i</sub>_). A joint represention is then learned for the two modailities, implemented by element-wise summation layer. In the final layer, Decoders translate one modality to another.
![X-Translator](https://user-images.githubusercontent.com/14010288/182496123-443ffd2c-278a-4900-8419-613da327bce2.jpg)

## Result
| Method | PDR+PA [1], [2], [4] | Vi-Fi [3] | ViTag |
| ------------- | ------------- |  ------------- |  ------------- |
| Avg. IDP | 38.41% | 82.93% | **88.39%** |
<img src="https://user-images.githubusercontent.com/14010288/182496241-0fe0adda-7897-46e4-815a-2b9da495a434.png" width="500">

## References
[1] J. C. Gower. Generalized procrustes analysis. Psychometrika, 40(1):33–51, 1975. <br/>
[2] W. Krzanowski. Principles of multivariate analysis, volume 23. OUP Oxford, 2000. <br/>
[3] H. Liu, A. Alali, M. Ibrahim, B. B. Cao, N. Meegan, H. Li, M. Gruteser, S. Jain, K. Dana, A. Ashok, et al. Vi-fi: Associating moving subjects across vision and wireless sensors. <br/>
[4] B. Wang, X. Liu, B. Yu, R. Jia, and X. Gan. Pedestrian dead reckoning based on motion mode recognition using a smartphone. Sensors, 18(6):1811, 2018.

---

# Code Instructions
## Environments
#### Ubuntu
```
18.04.1 LTS
```
#### NVIDIA
NVIDIA-SMI & Driver Version ```460.32.03```

CUDA Version ```11.2```

#### TensorFlow
```
tensorflow              2.3.0
tensorflow-estimator    2.3.0
tensorflow-gpu          2.3.0
```
#### Keras
```
Keras                   2.4.3
Keras-Applications      1.0.8
Keras-Preprocessing     1.1.2
```

## Conda Environment
```
conda create --name via --file via.txt
conda activate via
```

This repository contains two main steps: (1) [Data Conversion](https://github.com/bryanbocao/vitag/tree/main/src/data_converters) and (2) [Model](https://github.com/bryanbocao/vitag/tree/main/src/model). **(1) Data Conversion** preprocesses and synchronizes the raw data to prepare a synchronized dataset for **(2) Model**'s usage in a neat way. We provide the processed(RAN4model_dfv3) datasets so that skip the first part and run the second part of model scripts directly.

## Dataset for Model - dfv3
Download ```RAN4model``` [[Google Drive](https://drive.google.com/file/d/119aXAow_svc8NuIGbsXDoZgDTa_kyVzT/view?usp=sharing)] [[OneDrive](https://1drv.ms/u/s!AqkVlEZgdjnYi1NHD5LPJeSQ3XiO?e=vcx43N)] and follow the folder structure:
```
vitag
  |-Data
     |-checkpoints
     |-datasets
        |-RAN4model
  |-src
     |-...
```

## Training
__Indoor-scene0__ ```src/model/BBX5in_IMU19_FTM2/indoor```
```
cd src/model/BBX5in_IMU19_FTM2/indoor
```
Train from scratch
```
python3 train.py -l mse -tsid_idx 0
```
Resume training from last checkpoint
```
python3 train.py -l mse -tsid_idx 0 -rt
```

__Outdoor-scene1__ ```src/model/BBX5in_IMU19_FTM2/outdoor```
Train from scratch
```
python3 train_rand_ss_scene1.py -l mse -tt rand_ss -tsid_idx 0
```
Resume training from last checkpoint
```
python3 train_rand_ss_scene1.py -l mse -tt rand_ss -tsid_idx 0 -rt
```
where 
```
-l: training loss function of mse
-tsid_idx: training sequences by testing sequence id index(0-indexed). For instance, 
  when -tsid_idx 2, the sequence id with index 2 will be used for testing 
  while the remaining 14 sequences in the lab are used for training.
```

## Testing for Association

Note that Euclidean distance function(eucl) is used for all modalities by default.

### ViTag Models
Download [checkpoints](https://drive.google.com/drive/folders/1ETKgtK0Vs0Y8zCcEIfw8-2EyMOgHPIyL?usp=sharing) and follow the folder structure:
```
vitag
  |-Data
     |-checkpoints
       |-X22_indoor_BBX5in_IMU19_FTM2_test_idx_0
       |-...
       |-X22_outdoor_BBX5in_IMU19_FTM2_rand_ss_scene1_seq_0
       |-...
     |-datasets
  |-src
     |-...
```
Please add ```-bw``` to load these best weights to reproduce results.

__indoor__ ```src/model/BBX5in_IMU19_FTM2/indoor```
```
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -wd_ls 1 1 5 0 -bw -tsid_idx 0
```
__outdoor__ ```src/model/BBX5in_IMU19_FTM2/outdoor/eval_w_trained_scene1```

rand_ss & eucl(by default)
```
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -tt rand_ss -wd_ls 1 1 1 0 -bw -tsid_idx 0
```
rand_ss & bhattacharyya distance for FTM
```
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -tt rand_ss -wd_ls 1 1 1 0 -bw -f_d b -tsid_idx 0
```
crowded & eucl(by default)
```
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -tt crowded -wd_ls 1 1 1 0 -bw -tsid_idx 0
```
crowded & bhattacharyya distance for FTM
```
python3 eval_w_vis_save_pred.py -fps 10 -k 10 -tt crowded -wd_ls 1 1 1 0 -bw -f_d b -tsid_idx 0
```
where 
```
-fps: frame rate
-k: window length(# of frames in a window)
-wd_ls: list of weights of distances for different modalities in this order -- 
  0: BBX5, 1: IMU, 2: FTM, 3: D_FTM
-bw: load best weight, results can be reproduced provided weights
-tsid_idx: testing sequence index
-tt: test type of rand_ss or crowded.
-f_d: distance function for FTM, eucl(by default), b - bhattacharyya distance
```
### Result Interpretation
```
20201223_142404_f_d_eucl_12_28_21_12_08_49_w_ls_1_1_5_0_gd_cumu_Cam_IDP_0.4505_gd_cumu_Phone_IDP_0.6640_hg_cumu_Cam_IDP_0.6915_hg_cumu_Phone_IDP_0.9426
```
where
```
f_d_eucl: Euclidean distance is used for FTM modality.
w_ls: list of weights of distances for different modalities in this order -- 
  0: BBX5, 1: IMU, 2: FTM, 3: D_FTM
gd: Greedy-Matching
hg: Hungarian
cumu_Cam_IDP: Cumulative IDP in Camera domain
cumu_Phone_IDP: Cumulative IDP in Phone domain
```
We report ```hg_cumu_Phone_IDP``` in the paper. In the above example, we achieved IDP of ```0.9426``` in test sequenced index ```3``` in the ```indoor``` dataset.


### Baseline - Pedestrian Dead Reckoning + Procrustes Analysis (PDR + PA)
__indoor__ ```src/model/BBX5in_IMU19_FTM2/indoor```
```
python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0
```
Test with noise level ```-nl 0.1```
```
python3 eval_prct.py -fps 10 -k 10 -tsid_idx 0 -nl 0.1
```
__outdoor__ ```src/model/BBX5in_IMU19_FTM2/outdoor/eval_w_trained_scene1```

rand_ss
```
python3 eval_prct.py -fps 10 -tt rand_ss -k 10 -tsid_idx 2
```
```
python3 eval_prct.py -fps 10 -tt crowded -k 10 -tsid_idx 2
```
Test with noise level ```-nl 0.1```
```
python3 eval_prct.py -fps 10 -tt rand_ss -k 10 -tsid_idx 0 -nl 0.1
```

---

# Citation
ViTag BibTeX:
```
@inproceedings{cao2022vitag,
  title={ViTag: Online WiFi Fine Time Measurements Aided Vision-Motion Identity Association in Multi-person Environments},
  author={Cao, Bryan Bo and Alali, Abrar and Liu, Hansi and Meegan, Nicholas and Gruteser, Marco and Dana, Kristin and Ashok, Ashwin and Jain, Shubham},
  booktitle={2022 19th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)},
  pages={19--27},
  year={2022},
  organization={IEEE}
}
```

Vi-Fi (dataset) BibTex:
```
@inproceedings{liu2022vi,
  title={Vi-Fi: Associating Moving Subjects across Vision and Wireless Sensors},
  author={Liu, Hansi and Alali, Abrar and Ibrahim, Mohamed and Cao, Bryan Bo and Meegan, Nicholas and Li, Hongyu and Gruteser, Marco and Jain, Shubham and Dana, Kristin and Ashok, Ashwin and others},
  booktitle={2022 21st ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
  pages={208--219},
  year={2022},
  organization={IEEE}
}
```
```
@misc{vifisite,
  author        = "Hansi Liu",
  title         = "Vi-Fi Dataset",
  month         = "Dec. 05,", 
  year          = "2022 [Online]",
  url           = "https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home"
}
```

[Reality-Aware Networks Project Website](https://ashwinashok.github.io/realityawarenetworks/)

# Acknowledgement
This research has been supported by the National Science Foundation (NSF) under Grant Nos. CNS-2055520, CNS1901355, CNS-1901133. 
We thank Rashed Rahman,Shardul Avinash, Abbaas Alif, Bhagirath Tallapragada and Kausik Amancherla for their help with data labeling.
