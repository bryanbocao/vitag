import datetime
from csv import reader
from collections import defaultdict
'''
This method convert the Original TimeStamp (ots26) 2021-10-07 13:46:34.773741 into ts16_dfv3.
'''
def ots26_to_ts16_dfv3(ots26):
    year, month, day = int(ots26[:4]), int(ots26[5:7]), int(ots26[8:10])
    hour, minute, second = int(ots26[11:13]), int(ots26[14:16]), int(ots26[17:19])
    after_second = int(ots26[20:26])
    dt = datetime.datetime(year, month, day, hour, minute, second, after_second)

    # print();print() # Debug
    # print('dt.timestamp(): ', dt.timestamp())
    # e.g. dt.timestamp():  1633628794.773

    # Debug
    dt_ts_str = str(dt.timestamp()) # ts16_dfv3
    while len(dt_ts_str) < 17:
        dt_ts_str += '0'
    # print('dt_ts_str: ', dt_ts_str, ', len(dt_ts_str): ', len(dt_ts_str))
    # e.g. dt_ts_str:  1633628794.773000 , len(dt_ts_str):  17
    ts16_dfv3 = dt_ts_str
    return ts16_dfv3

'''
# debug
ots26 = '2021-10-07 13:46:34.773000'
ts16_dfv3 = ots26_to_ts16_dfv3(ots26)
print('ts16_dfv3: ', ts16_dfv3, ', len(ts16_dfv3): ', len(ts16_dfv3))
# e.g. ts16_dfv3:  1633628794.773000 , len(ts16_dfv3):  17
'''


def ts16_dfv3_to_ots26(ts16_dfv3):
    dt = datetime.datetime.fromtimestamp(int(ts16_dfv3[:ts16_dfv3.index('.')]))
    # print(); print()
    # print('dt: ', dt)
    # e.g. 2021-10-07 13:46:34
    ots26 = dt.strftime("%Y-%m-%d %H:%M:%S") + '.' + ts16_dfv3[ts16_dfv3.index('.') + 1:]
    # print('ots26: ', ots26, ', len(ots26): ', len(ots26))
    # e.g. 2021-10-07 13:46:34.773000 26
    return ots26
'''
# debug
ts16_dfv3 = '1633628794.773000'
ts16_dfv3_to_ots26(ts16_dfv3)
'''

def get_start_end_ts_update_phone_with_offsets(seq_id, RAN_data_status_path):
    with open(RAN_data_status_path, 'r') as f:
        csv_reader = reader(f)
        for i, row in enumerate(csv_reader):
            if row[0] == seq_id:
                print(); print() # debug
                print(row)
                '''
                e.g.
                ['20211007_133041', 'Bo', 'uploaded', 'Labeled', 'Yes',
                '2021-10-07 13:31:05.024299, 2021-10-07 13:34:05.234398',
                'Hansi: -1800.000000\nNicholas: 3300.000000\nBo: 1300.000000']
                '''
                start_ots, end_ots = row[5].split(', ')
                start_ts16_dfv3, end_ts16_dfv3 = ots26_to_ts16_dfv3(start_ots), ots26_to_ts16_dfv3(end_ots)
                print(); print() # debug
                print('start_ots: ', start_ots, ', end_ots: ', end_ots)
                print('start_ts16_dfv3: ', start_ts16_dfv3, ', end_ts16_dfv3: ', end_ts16_dfv3)
                '''
                e.g.
                start_ots:  2021-10-07 13:31:05.024299 , end_ots:  2021-10-07 13:34:05.234398
                start_ts16_dfv3:  1633627865.024299 , end_ts16_dfv3:  1633628045.234398
                '''
                print(); print() # debug
                print('row[-1].split(): ', row[-1].split('\n')) # e.g. ['Hansi: -1800.000000', 'Nicholas: 3300.000000', 'Bo: 1300.000000']
                subj_to_offset_ls = row[-1].split('\n')
                subj_to_offset = defaultdict()
                for subj_offset in subj_to_offset_ls:
                    subj, offset = subj_offset.split(': ')
                    subj_to_offset[subj] = float(offset) / 1000 # sec
                print(); print() # debug
                print(subj_to_offset)
                '''
                e.g.
                defaultdict(None, {'Hansi': -1800.0, 'Nicholas': 3300.0, 'Bo': 1300.0})
                '''
                return start_ots, end_ots, subj_to_offset
    return None
