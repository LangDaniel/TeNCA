from pathlib import Path
import pickle
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py

class GetPatches():

    def __init__(
        self,
        bbox_file,
        file_path,
        spacing,
        identifier='identifier',
        orientation='LPS',
        bbox_cols=[
            'L_min',
            'L_max',
            'P_min',
            'P_max',
            'S_min',
            'S_max',
        ],
        instances=['path'],
    ):

        # merge tissue bbox file with path file
        # in the case_df data frame
        # -------------------------------------------------
        bbox_df = pd.read_csv(bbox_file)
        self.bbox_cols = bbox_cols
        if isinstance(file_path, pd.core.frame.DataFrame):
            path_df = file_path
        else:
            path_df = pd.read_csv(file_path)
        path_df = path_df[[identifier]+instances].drop_duplicates()
        self.instances = instances
        case_df = bbox_df.merge(
            path_df,
            on=identifier,
            how='left',
        )
        self.case_df = case_df.set_index(identifier)
        # -------------------------------------------------

        self.spacing = np.array(spacing)
        self.identifier = identifier
        self.orientation = orientation
    # ----------------------------------------------------------------------------------
    # Read data
    # ----------------------------------------------------------------------------------
    def get_itk_from_path(self, path):
        path = Path(path)
        if path.is_dir():
            return self.read_dcm_folder(path, self.orientation)
        if (path.suffix == '.gz') or (path.suffix == '.nii'):
            return self.read_nii_file(path, self.orientation)

        raise ValueError('unknown file format')

    @staticmethod
    def read_dcm_folder(path, orientation):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)

        data = reader.Execute()
        if orientation:
            data = sitk.DICOMOrient(data, orientation)

        return data

    @staticmethod
    def read_nii_file(path, orientation='LPS'):
        data = sitk.ReadImage(str(path))
        if orientation:
            data = sitk.DICOMOrient(data, orientation)

        return data

    @staticmethod
    def get_array_from_itk(image):
        data = sitk.GetArrayFromImage(image)
        return np.moveaxis(data, 0, -1)

    def get_acquisition_times(self, pid):
        times = literal_eval(self.acq_df.loc[pid]['acquisition_times'])
        return np.array(times, dtype=int)
    # ----------------------------------------------------------------------------------
    # bbox stuff
    # ----------------------------------------------------------------------------------
    def get_bbox_LPS(self, pid):
        row = self.case_df.loc[pid]
        bbox = row[self.bbox_cols].values
        return bbox

    def get_bbox_RCS(self, ds, bbox_LPS):
        bbox_RCS = np.zeros(6)
        bbox_RCS[::2] = ds.TransformPhysicalPointToIndex(
            bbox_LPS[::2]
        )
        bbox_RCS[1::2] = ds.TransformPhysicalPointToIndex(
            bbox_LPS[1::2]
        )
        bbox_RCS =  bbox_RCS.astype(int)
        return bbox_RCS

    @staticmethod
    def crop_patch(data, bbox):
        xii, xff, yii, yff, zii, zff = bbox.astype(int)
        return data[xii:xff, yii:yff, zii:zff]
    # ----------------------------------------------------------------------------------
    # SITK image manipulation
    # ----------------------------------------------------------------------------------
    def resample_img(self, itk_image, is_label=False):

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        spacing = np.zeros(3)
        for ii in range(0, 3):
            if self.spacing[ii] == False:
                spacing[ii] = original_spacing[ii]
            else:
                spacing[ii] = self.spacing[ii]

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        if is_label:
            resample.SetDefaultPixelValue(0)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    def padding(self, itk_img, bbox, const='min'):
        lower_pad = itk_img.TransformPhysicalPointToIndex(bbox[::2])
        lower_pad = np.clip(np.zeros(3) - lower_pad, a_min=0, a_max=None).astype(int)

        upper_pad = np.array(itk_img.TransformPhysicalPointToIndex(bbox[1::2]))
        size = np.array(itk_img.GetSize())
        upper_pad = np.clip(upper_pad-size, a_min=0, a_max=None).astype(int)

        if not lower_pad.any() and not upper_pad.any():
            return itk_img
        print('[INFO]: padding')

        # convert to list due to sitk bug
        lower_pad = lower_pad.astype('int').tolist()
        upper_pad = upper_pad.astype('int').tolist()

        # select padding values
        # ---------------------
        if isinstance(const, (int, float)):
            constant = const
        elif isinstance(const, str):
            statsf = sitk.StatisticsImageFilter()
            statsf.Execute(itk_img)
            if const == 'min':
                constant = statsf.GetMinimum()
            elif const == 'max':
                constant = statsf.GetMaximum()
            elif const == 'mean':
                constant = statsf.GetMean()
            else:
                raise ValueError(f'unknown const value type: {const}')
        else:
            raise ValueError(f'unknown const value type: {const}')
        # ---------------------
        return sitk.ConstantPad(itk_img, lower_pad, upper_pad, constant=constant)
    # ----------------------------------------------------------------------------------
    # helper functions for normalization
    # ----------------------------------------------------------------------------------
    @staticmethod
    def rescale_linear(data, scaling, vmin=None, vmax=None):
        assert type(vmin) in [type(None), float, int, bool]
        assert type(vmax) in [type(None), float, int, bool]
        [src_min, src_max], [dst_min, dst_max] = scaling

        slope = (dst_max - dst_min) / (src_max - src_min)
        bias = dst_min - slope * src_min
        data = slope * data + bias
        if isinstance(vmin, (float, int)) or isinstance(vmax, (float, int)):
            data = np.clip(a=data, a_min=vmin, a_max=vmax)
        return data

    def normalize_image(self, img, mean, std):
        dst_mean, dst_std = mean, std
        f = sitk.StatisticsImageFilter()
        f.Execute(img)
        src_mean = f.GetMean()
        src_std = f.GetVariance()**(1/2)
        return (((img - src_mean) / src_std)*dst_std)+dst_mean
    # ----------------------------------------------------------------------------------
    # get data
    # ----------------------------------------------------------------------------------
    def get_instance(self, pid, instance):
        path = self.case_df.loc[pid][instance]

        try:
            # read and reample image
            ds = self.get_itk_from_path(path)
            bbox_LPS = self.get_bbox_LPS(pid)
            if self.spacing.any():
                ds = self.resample_img(ds, is_label=(instance=='sgmt'))
            ds = self.padding(ds, bbox_LPS, const='min')
            # crop using bbox
            bbox_RCS = self.get_bbox_RCS(ds, bbox_LPS)
            ds = self.crop_patch(ds, bbox_RCS)
            # convert to numpy

            data = self.get_array_from_itk(ds)
        except (Exception, ArithmeticError) as e:
            raise ValueError(f'{pid}-{instance} failed: {e}')

        return data

    def get_case(self, pid):
        raise NotImplementedError
        #data = {}
        #for inst in self.instances:
        #    if self.case_df.loc[pid][inst] == 'False':
        #        continue
        #    data[f'{pid}/{inst}'] = self.get_instance(pid, inst)

        #vmax = np.percentile(list(data.values()), 99.98)
        #vmin = np.percentile(list(data.values()), 0.02)
        #scaling = [[vmin, vmax], [0., 1.]]
        #for key, value in data.items():
        #    data[key] = self.rescale_linear(value, scaling, True)

        #return data

    def to_disk(self, path):
        # prep output file and folder
        # ----------------------------------------
        path = Path(path)
        if path.exists():
            raise ValueError('output file exists')
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        # ----------------------------------------
        # save identifiers to csv
        disk_df = pd.DataFrame(
            columns=[self.identifier]+self.instances
        )
        # ----------------------------------------

        with h5py.File(path, 'w') as ff:
            for pid, row in self.case_df.iterrows():
                print(f'{pid}')
                try:
                    data = self.get_case(pid)
                    for name, array in data.items():
                        ff.create_dataset(
                            name,
                            data=array
                        )

                except (Exception, ArithmeticError) as e:
                    print(f'failed: {e}')
                    continue

                exists = [(f != 'False') for f in row[self.instances]]
                disk_df.loc[len(disk_df)] = [pid, *exists]

        disk_df.to_csv(
            path.parent / (path.stem + '.csv'),
            index=False
        )
