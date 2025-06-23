from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np

def read_image(path, orientation):
    path = Path(path)

    if path.suffix == '.gz' or path.suffix == '.nii':
        return read_nifti(path, orientation)
    elif path.isdir():
        return read_dicom(path, orientation)
    else:
        raise ValueError('unknown format')

def read_dicom(path, orientation=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)

    data = reader.Execute()
    if orientation:
        data = sitk.DICOMOrient(data, orientation)

    return data

def read_nifti(path, orientation=False):
    data = sitk.ReadImage(str(path))
    if orientation:
        data = sitk.DICOMOrient(data, orientation)

    return data

def array_from_itk(image):
    data = sitk.GetArrayFromImage(image)
    data = np.moveaxis(data, 0, -1)
    return data

def get_bbox_CRS(sgmt):
    bbox_RCS = np.zeros(6)
    contour = np.where(sgmt)
    bbox_RCS[::2] = np.min(contour, axis=1)
    bbox_RCS[1::2] = np.max(contour, axis=1)

    # RCS to CRS
    bbox_CRS = bbox_RCS.copy()
    bbox_CRS[0:2] = bbox_RCS[2:4]
    bbox_CRS[2:4] = bbox_RCS[0:2]
    return bbox_CRS.astype(int)

file_df = pd.read_csv('./../../data/file_path/mama_duke_file_path.csv')

orientation = 'LPS'
if not orientation:
    columns = ['identifier', 'ax1_min', 'ax1_max', 'ax2_min', 'ax2_max', 'ax3_min', 'ax3_max']
else:
    columns = ['identifier']
    for ax in orientation:
        columns += [f'{ax}_min', f'{ax}_max']

bbox_df = pd.DataFrame(columns=columns)

for idx, row in file_df.iloc[:].iterrows():
    pid = row['identifier']
    sgmt_ds = read_image(row['sgmt'], orientation)
    sgmt = array_from_itk(sgmt_ds)
    # bbox column, rows, slices
    bbox_CRS = get_bbox_CRS(sgmt)
    # bbox LPS
    bbox_LPS = np.zeros(6)
    bbox_LPS[::2] = sgmt_ds.TransformIndexToPhysicalPoint(
        list(bbox_CRS[::2].astype('int').tolist())
    )
    bbox_LPS[1::2] = sgmt_ds.TransformIndexToPhysicalPoint(
        list(bbox_CRS[1::2].astype('int').tolist())
    )
    bbox_df.loc[len(bbox_df)] = [pid, *bbox_LPS]

bbox_df.to_csv(f'./../../data/bboxes/lesion_bboxes_orientation_{str(orientation)}.csv', index=False)
