from ast import literal_eval
from pathlib import Path
import patch_generator
import numpy as np
import pandas as pd

# set parameter
# ------------------------------------
bbox_file = './../../data/bboxes/lesion_bboxes_orientation_LPS_size_168x168x64.csv'
file_path = './../../data/file_path/mama_duke_file_path.csv'
spacing = [1., 1., 1.]
orientation = 'LPS'

upper_percentile = 99.98
lower_percentile = 0.02

output_file = f'/home/iml/lang/Projects/CoMo-NCA/data/patches/mama_duke_lps_168x168x64_linrescale_pre_scaling/patches.h5'
# ------------------------------------
# prep acquisition times
acq_df = pd.read_csv('./../../data/meta_info/mama_duke_acquisition_times.csv')

acq_dict = {}
for i, row in acq_df.iterrows():
    try:
        times = row['acquisition_times']
        if pd.isna(times):
            continue
        times = np.array(
            literal_eval(times),
            dtype=int
        )
        acq_dict[row['identifier']] = times
    except:
        pid = row['identifier']
        print(f'failed: {i}-{pid}')
# ------------------------------------

pg = patch_generator.GetPatches(
    bbox_file=bbox_file,
    file_path=file_path,
    spacing=spacing,
    orientation=orientation,
    instances=['pre']+[f'post_{i}' for i in range(0, 5)]+['sgmt']
)
#def get_case(pid):
#    data = {}
#    for inst in pg.instances:
#        if pg.case_df.loc[pid][inst] == 'False' or inst == 'sgmt':
#            continue
#        data[f'{pid}/{inst}'] = pg.get_instance(pid, inst)
#
#    # rescale batch based on upper and lower percentile
#    vmax = np.percentile(
#        list(data.values()),
#        upper_percentile
#    )
#    vmin = np.percentile(
#        list(data.values()),
#        lower_percentile
#    )
#    scaling = [[vmin, vmax], [0., 1.]]
#    for key, value in data.items():
#        data[key] = pg.rescale_linear(value, scaling, True)
#    # -------------------------------------------------
#    # add sgmt
#    data[f'{pid}/sgmt'] = pg.get_instance(pid, 'sgmt')
#    # -------------------------------------------------
#    # add acquisition times
#    if pid in acq_dict:
#        data[f'{pid}/acquisition_times'] = acq_dict[pid]
#
#    return data

def get_case(pid):
    data = {}
    pre = pg.get_instance(pid, 'pre')
    data[f'{pid}/pre'] = pre

    post_inst = [i for i in pg.instances if i.startswith('post_')]
    for inst in post_inst:
        if pg.case_df.loc[pid][inst] == 'False':
            continue
        data[f'{pid}/{inst}'] = pg.get_instance(pid, inst)

    # rescale batch based on upper and lower percentile of pre image
    # and clip min values at 0.
    vmax = np.percentile(pre, upper_percentile)
    vmin = np.percentile(pre, lower_percentile)
    scaling = [[vmin, vmax], [0., .25]]
    for key, array in data.items():
        data[key] = pg.rescale_linear(data=array, scaling=scaling, vmin=0.)
    # -------------------------------------------------
    # add sgmt
    data[f'{pid}/sgmt'] = pg.get_instance(pid, 'sgmt')
    # -------------------------------------------------
    # add acquisition times
    if pid in acq_dict:
        data[f'{pid}/acquisition_times'] = acq_dict[pid]

    return data

pg.get_case = get_case

# testing
# pg.case_df = pg.case_df.iloc[:3]

pg.to_disk(output_file)

# save scripts for reproducibility
for f in [__file__, patch_generator.__file__]:
    content = Path(f).read_text()
    path = Path(output_file).parent / Path(f).name
    with open(path, 'w') as dst:
        dst.write(content)
