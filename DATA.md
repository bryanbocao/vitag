# Data

#### News:
01/16/2024 We released the synchronized version (**RAN4model_dfv4p4**) of our data for future usage:

| | Google Drive | OneDrive|
|--|--|--|
| RAN4model_dfv4p4.zip | [link](https://drive.google.com/file/d/1tt458YclrKMMGG-P6lupuGAaN09bTsht/view?usp=sharing) | [link](https://1drv.ms/u/s!AqkVlEZgdjnYlSNt6e62k0jAl9x2?e=ZzzEMZ) |
| RAN4model_dfv4p4 folder | [link](https://drive.google.com/drive/folders/1oZz2F7HmXwjV8rvS6I2H87ebL_MQwYy5?usp=sharing) | [link](https://1drv.ms/f/s!AqkVlEZgdjnYlRWGHeAAawAlaFCR?e=qketyS) |

**RAN4model_dfv4p4** provides you with the convenient synchronized format for downstream tasks. In this document, we take one subject in scene4 from one outdoor sequence as an example to demonstrate the format.

## Folder Structure
```
RAN4model_dfv4p4
  |-exps
    |-exp1
      |-start_end_ts16_dfv4p4_indoor.json
      |-start_end_ts16_dfv4p4_outdoor.json
  |-seqs
    |-indoor
      |-scene0
        |-<seq_id>
        |-20201223_140951
        |-20201223_141456
        |-20201223_141923
          |-RGBg_ts16_dfv4p4_ls.json
          |-RGBh_ts16_dfv4p4_ls.json
            |-sync_ts16_dfv4p4
              |-BBX5H_sync_dfv4p4.pkl
              |-BBXC3H_sync_dfv4p4.pkl
              |-FTM_sync_dfv4p4.pkl
              |-FTM_li_sync_dfv4p4.pkl
              |-IMU19_sync_dfv4p4.pkl
              |-IMUagm9_sync_dfv4p4.pkl
              |-RSSI_sync_dfv4p4.pkl
              |-RSSI_li_sync_dfv4p4.pkl
              ...
        ...
    |-outdoor
      |-scene1
      |-scene2
      |-scene3
      |-scene4
        |-20211007_135415
        |-20211007_135800
        |-20211007_135010
          |-RGBg_ts16_dfv4p4_ls.json
          |-RGB_ts16_dfv4p4_ls.json
          |-sync_ts16_dfv4p4
            |-BBX5_sync_dfv4p4.pkl
            |-BBX5_Others_sync_dfv4p4.pkl
            |-BBXC3_sync_dfv4p4.pkl
            |-BBXC3_Others_sync_dfv4p4.pkl
            |-FTM_sync_dfv4p4.pkl
            |-FTM_li_sync_dfv4p4.pkl
            |-IMU19_sync_dfv4p4.pkl
            |-IMUagm9_sync_dfv4p4.pkl
            |-RSSI_sync_dfv4p4.pkl
            |-RSSI_li_sync_dfv4p4.pkl
            |-Others_id_ls.pkl
       ...
```

## Data Loading
```
import pickle as pkl
with open(<M_sync_dfv4p4_path>, 'rb') as f:
    M_sync_dfv4p4 = pkl.load(f); print(f'{M_sync_dfv4p4_path} loaded!')
```
where ```M``` stands for ```modality```. You may replace ```M``` with any modality such as ```BBX5_sync_dfv4p4```.

When loading the json file, in case of the ```not JSON serializable``` error, add the following lines of code:
```
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
```

# Vi-Fi Dataset
[Official Dataset link](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)

[paperswithcode link](https://paperswithcode.com/dataset/vi-fi-multi-modal-dataset)
