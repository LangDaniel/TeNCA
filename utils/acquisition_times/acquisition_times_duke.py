import dicom_reader as rdr
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_tag(path, tag):
    path = str(path)
    ds = rdr.DICOMImage(path)
    return ds.get_tag(tag)

def tm_to_dt(tm_str, format_string='%H%M%S'):
    # discard milliseconds
    tm_str = tm_str.split('.')[0]
    # check a max length of six digits
    if len(tm_str) > 6:
        raise ValueError('format error string longer than 6 digits')
    # secure lenght of six
    tm_str = f'{int(tm_str):06d}'
    return datetime.strptime(tm_str, format_string)

file_df = pd.read_csv('./../../data/file_path/duke_file_path.csv')
sequences = list(file_df)[1:-1]

tag = 'AcquisitionTime'
time_dict = {}
for idx, row in file_df.iterrows():
    times = [0]
    start_time = tm_to_dt(get_tag(row[sequences[0]], tag))
    for seq in sequences[1:]:
        path = row[seq]
        if path != 'False' and path != False:
            time = tm_to_dt(get_tag(path, tag))
            times += [int((time-start_time).seconds)]
    time_dict[row['identifier']] = times

time_df = pd.DataFrame(columns=['identifier', 'acquisition_times'])
for key, value in time_dict.items():
    pid = 'DUKE_' + key[-3:]
    time_df.loc[len(time_df)] = [pid, value]

time_df.to_csv('./../../data/meta_info/duke_acquisition_times.csv', index=False)
