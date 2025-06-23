from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

mama_df = pd.read_csv('/mnt/IML-Proj/public_datasets/MAMA-MIA/train_test_splits.csv')
time_df = pd.read_csv('./../../data/meta_info/mama_duke_acquisition_times.csv')

train_df = mama_df[['train_split']].reset_index(drop='True')
train_df = train_df.rename(columns={'train_split': 'identifier'})
# remove cases with no time info
train_df = train_df[train_df['identifier'].isin(time_df.identifier)].reset_index(drop=True)

test_df = mama_df[~mama_df['test_split'].isna()][['test_split']].reset_index(drop='True')
test_df = test_df.rename(columns={'test_split': 'identifier'})
# remove cases with no time info
test_df = test_df[test_df['identifier'].isin(time_df.identifier)].reset_index(drop=True)

# add duke cases not in mama mia to train set
mama_ids = list(train_df.identifier.values) + list(test_df.identifier.values)
duke_df = time_df[~time_df['identifier'].isin(mama_ids)][['identifier']]
train_df = pd.concat([train_df, duke_df]).reset_index(drop=True)

assert not np.any(np.intersect1d(test_df['identifier'].values, train_df['identifier'].values))

train_df['cohort'] = train_df['identifier'].apply(lambda x: x.split('_')[0])

cohort_dict = {k: v for v, k in enumerate(train_df['cohort'].unique())}
train_df['cohort_num'] = train_df['cohort'].map(cohort_dict)

split_size = 200
split_gen = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)

for train_idx, valid_idx in split_gen.split(train_df, train_df['cohort_num']):
    valid_df = train_df.loc[valid_idx].copy().reset_index(drop=True)
    train_df = train_df.loc[train_idx].copy().reset_index(drop=True)

assert not np.any(np.intersect1d(valid_df['identifier'].values, train_df['identifier'].values))

to_disk = True
root_dir = Path('./../../data/labels/mama_duke_with_time')

if to_disk:
    root_dir.mkdir(parents=True)
    train_df.to_csv(root_dir / 'train.csv', index=False)
    valid_df.to_csv(root_dir / 'valid.csv', index=False)
    test_df.to_csv(root_dir / 'test.csv', index=False)
