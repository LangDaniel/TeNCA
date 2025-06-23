from pathlib import Path
import pandas as pd

mama_img_root_dir = Path('/mnt/IML-Proj/public_datasets/MAMA-MIA/images/')
mama_seg_root_dir = Path('/mnt/IML-Proj/public_datasets/MAMA-MIA/segmentations/expert/')

mama_img_cases = [f for f in mama_img_root_dir.iterdir() if f.is_dir()]

mama_case_dict = {}
max_length = 0
for ca in mama_img_cases:
    files = sorted([f for f in ca.iterdir() if f.suffix == '.gz'])
    mama_case_dict[ca.stem] = files
    if len(files) > max_length:
        max_length = len(files)

columns = ['identifier'] + ['pre'] + [f'post_{i}' for i in range(0, max_length-1)]
mama_df = pd.DataFrame(columns=columns)

for pid, files in mama_case_dict.items():
    paths = [False]*max_length
    paths[:len(files)] = files
    mama_df.loc[len(mama_df)] = [pid, *paths]

mama_seg_files = [f for f in mama_seg_root_dir.iterdir() if f.suffix == '.gz']

mama_seg_df = pd.DataFrame(columns=['identifier', 'sgmt'])
for f in mama_seg_files:
    pid = f.name.split('.')[0]
    mama_seg_df.loc[len(mama_seg_df)] = [pid, str(f)]

mama_df = mama_df.merge(mama_seg_df)
mama_df.to_csv('./../../data/file_path/mama_file_path.csv', index=False)

duke_img_root_dir = Path('/mnt/IML-Proj/public_datasets/TCIA/Duke-Breast-Cancer-MRI/images/')
duke_seg_root_dir = Path('/mnt/IML-Proj/public_datasets/TCIA/Duke-Breast-Cancer-MRI/segmentations/gtv/')

duke_cases = [f for f in duke_img_root_dir.iterdir() if f.is_dir()]

duke_case_dict = {}
for ca in duke_cases:
    files = sorted([f for f in ca.iterdir() if f.stem.startswith('pre')])
    files += sorted([f for f in ca.iterdir() if f.stem.startswith('post')])
    duke_case_dict[ca.stem] = files
    if len(files) > max_length:
        max_length = len(files)

columns = ['identifier'] + ['pre'] + [f'post_{i}' for i in range(0, max_length-1)]
duke_df = pd.DataFrame(columns=columns)

for pid, files in duke_case_dict.items():
    paths = [False]*max_length
    paths[:len(files)] = files
    duke_df.loc[len(duke_df)] = [pid, *paths]

duke_seg_files = [f for f in duke_seg_root_dir.iterdir() if f.suffix == '.gz']

duke_seg_df = pd.DataFrame(columns=['identifier', 'sgmt'])
for f in duke_seg_files:
    pid = f.name.split('.')[0]
    duke_seg_df.loc[len(duke_seg_df)] = [pid, str(f)]

duke_df = duke_df.merge(duke_seg_df)
duke_df.to_csv('./../../data/file_path/duke_file_path.csv', index=False)

# remove Duke cases already present in MAMA-MIA
duke_in_mama_ids = mama_df[mama_df.identifier.str.startswith('DUKE')]['identifier'].values
duke_df['identifier'] = 'DUKE_' + duke_df.identifier.str[-3:]
duke_df = duke_df[~duke_df['identifier'].isin(duke_in_mama_ids)]

file_df = pd.concat([mama_df, duke_df]).reset_index(drop=True)
file_df.to_csv('./../../data/file_path/mama_duke_file_path.csv', index=False)
