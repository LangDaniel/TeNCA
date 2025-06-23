from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, args, rtn_idx=False):
        self.args = args
        # set fixed seed
        np.random.seed(42)

        patch_file = Path(args['data_root']) / args['patch_file']
        disk_df = pd.read_csv(patch_file.parent / (patch_file.stem + '.csv'))
        self.patch_file = patch_file

        label_df = pd.read_csv(Path(args['data_root']) / args['label_file'])
        case_df = label_df.merge(disk_df, how='inner', on='identifier')
        # remove cases which do not feature post sequences following the time rules
        pids_with_times = self.check_times(case_df['identifier'].values)
        case_df = case_df[case_df['identifier'].isin(pids_with_times)].reset_index(drop=True)

        # add slice numbers to case_df
        if type(args['slices']) == list:
            self.case_df = pd.DataFrame(columns=list(case_df)+['slice'])
            for idx, row in case_df.iterrows():
                for slz in self.args['slices']:
                    self.case_df.loc[len(self.case_df)] = [*row.values, slz]
        else:
            print('reading slice info from: ', args['slices'])
            slice_df = pd.read_csv(Path(args['data_root'])/args['slices'])
            self.case_df = pd.merge(case_df, slice_df)

        # sample (train) or sort (val)
        if self.args['training']:
            self.case_df = self.case_df.sample(frac=1, random_state=42)
        else:
            self.case_df = self.case_df.sort_values(by=['identifier'])
        self.case_df = self.case_df.reset_index(drop=True)

        self.total_count = len(self.case_df)
        if not self.total_count:
            raise ValueError('no cases were found: check label naming')
        print(f'len {self.total_count}')

        sequences = []
        for seq in args['sequences']:
            if not seq[-1] == '_':
                sequences += [seq]
            else:
                sequences += [f'{seq}{i}' for i in range(args['n_time_points'])]
        self.args['sequences'] = sequences
        self.rtn_idx = rtn_idx

    def check_times(self, identifiers):
        '''checks if a case fulfills the requirements to be included,
        i.e. if it features timepoints smaller max_time'''
        has_time = []
        for pid in identifiers:
            times = self.get_sequence(pid, 'acquisition_times')
            times = times[1:self.args['n_time_points']+1]
            times = times[times <= self.args['max_time']]
            if times.any():
                has_time.append(pid)
            else:
                print('removing', pid)
        return has_time


    # augmentation
    # ----------------------------------------------------------------
    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high], [out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def add_gaussian_noise(data, noise_var):
        variance = np.random.uniform(noise_var[0], noise_var[1])
        data = data + np.random.normal(0.0, variance, size=data.shape)

        return data

    @staticmethod
    def gaussian_blur(data, sigma):
        sig = np.random.uniform(sigma[0], sigma[1])
        return gaussian_filter(data, sig)

    @staticmethod
    def add_offset(data, var):
        off = np.random.uniform(var[0], var[1])
        data = data + off

        return data

    @staticmethod
    def resample(data, xy_zoom, z_zoom):
        fac = [xy_zoom, xy_zoom, z_zoom]

        return zoom(data, zoom=fac, order=1)

    def get_aug_func(self):
        ''' returns a list with all the augmentations '''
        aug = []
        if self.args['augment']['flip']:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=0)
                )
            # flip on coronal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=1)
                )

        if self.args['augment']['rot']:
            # rotate k*90 degree
            rot_k = np.random.randint(0, 4)
            if rot_k:
                aug.append(
                    lambda data: np.rot90(
                        data, axes=(0, 1), k=rot_k
                    )
                )

        if self.args['augment']['zoom']:
            xy_zoom = np.random.uniform(
                self.args['augment']['zoom'][0],
                self.args['augment']['zoom'][1],
            )
            z_zoom = np.random.uniform(
                1.,
                self.args['augment']['zoom'][1],
            )
            aug.append(
                lambda data: self.resample(
                    data,
                    xy_zoom,
                    z_zoom,
                )
            )

        if self.args['augment']['offset']:
            aug.append(
                lambda data: self.add_offset(
                    data,
                    self.args['augment']['offset']
                )
            )

        if self.args['augment']['blur']:
            if self.args.augment.blur.type.lower() == 'gaussian':
                aug.append(
                    lambda data: self.gaussian_blur(
                        data,
                        self.args['augment']['blur']['sigma']
                    )
                )
            else:
                raise ValueError('blurring not registered')

        if self.args['augment']['noise']:
            if self.args['augment']['noise']['type'].lower() == 'gaussian':
                aug.append(
                    lambda data: self.add_gaussian_noise(
                        data,
                        self.args['augment']['noise']['variance']
                    )
                )
            else:
                raise ValueError('noise not registered')

        return aug

    def augmentation(self, data, aug_func):
        for func in aug_func:
            data = func(data)
        return data
    # ----------------------------------------------------------------
    # get data
    # ----------------------------------------------------------------

    @staticmethod
    def to_slice(shape):
        slz = []
        for shp in shape:
            slz.append(np.s_[:shp])
        return tuple(slz)

    def zero_pad(self, img, shape):
        # zero pad if too small
        padw = np.zeros(6)
        for ax in range(img.ndim):
            diff = shape[ax] - img.shape[ax]
            if diff < 0:
                continue
            padw[::2][ax] = int(np.ceil(diff/2))
            padw[1::2][ax] = diff//2
        padw = padw.reshape(3, 2).astype(int)
        if padw.any():
            img = np.pad(img, pad_width=padw)
        return img

    def assert_size(self, img):
        img = img[self.to_slice(self.args['shape'])]
        return img

    def get_sequence(self, pid, sequence, slize=np.s_[:]):
        with h5py.File(self.patch_file, 'r') as ff:
            data = ff[f'{pid}'][sequence][slize]

        return data

    def get_time_points(self, pid):
        times = self.get_sequence(pid, 'acquisition_times')
        # remove zero time point
        times = times[1:self.args['n_time_points']+1]
        # remove elements larger than max_time
        times = times[times <= self.args['max_time']]
        # add random noise with: low, high = args['time_noise']
        if self.args['training'] and np.any(self.args['time_noise']):
            low, high = self.args['time_noise']
            # set noise to be lower than time delta to not destroy order
            if len(times) > 1:
                max_diff = max(np.diff(times)) - 1
                low = max([-max_diff, low])
                high = min([max_diff, high])
            # add noise
            times += np.random.randint(low, high, len(times))
            # clip to stay within bounds
            times = np.clip(times, a_min=0, a_max=self.args['max_time'])
        # zero pad
        #if self.args['pad_outputs'] and len(times) < self.args['n_time_points']:
        times_ = np.zeros(self.args['n_time_points'])
        times_[:len(times)] = times
        times = times_

        return times.astype(int)

    def get_data(self, idx):
        row = self.case_df.loc[idx]
        pid = row['identifier']

        times = self.get_time_points(pid)

        slz = np.s_[-168:, -168:, row['slice']:row['slice']+1]
        data = []
        for seq in self.args['sequences']:
            # continue if the case misses the sequence
            if not bool(row[seq]):
                continue
            # omitt post sequences aquired after max_time
            if seq.startswith('post_'):
                n = int(seq[-1])
                if not times[n]:
                    continue
            data += [self.get_sequence(pid, seq, slz)]
            #if self.args['input_channels'] == 3:
            #    data += [self.get_sequence(pid, seq)[-224:, -224:, 31:34]]
            #elif self.args['input_channels'] == 1:
            #    data += [self.get_sequence(pid, seq)[-224:, -224:, 32:33]]
            #else:
            #    raise ValueError('unknown number of input channels')
        inputs, *outputs, sgmt = data
        outputs = np.array(outputs)
        if self.args['subtraction']:
            for i in range(len(outputs)):
                outputs[i] = np.clip((outputs[i] - inputs), a_min=0., a_max=None)
        if self.args['set_input_as_min']:
            for i in range(len(outputs)):
                outputs[i] = (inputs + np.clip((outputs[i] - inputs), a_min=0., a_max=None))

        # pad with nan to desired size if pad_outputs is set
        if self.args['pad_outputs'] and (len(outputs) < self.args['n_time_points']):
            shape = outputs.shape
            outputs_ = np.empty((self.args['n_time_points'], *shape[1:]))
            outputs_[:] = np.nan
            outputs_[:shape[0]] = outputs
            outputs = outputs_
        return inputs, outputs, sgmt.astype(np.float32), times
    # ----------------------------------------------------------------
    # torch Dataset stuff
    # ----------------------------------------------------------------
    def __len__(self):
        ''' cases involved '''
        return self.total_count

    def __getitem__(self, idx, rtn_idx=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs, outputs, sgmt, times = self.get_data(idx)
        #print(idx, inputs.shape, outputs.shape, sgmt.shape, times.shape)

        if self.rtn_idx:
            return inputs, outputs, sgmt, times, idx
        return inputs, outputs, sgmt, times
    # ----------------------------------------------------------------
