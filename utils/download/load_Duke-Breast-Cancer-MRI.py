from TCIA_client import TCIAClient
from pathlib import Path
import pandas as pd

base_url = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query'
client = TCIAClient(base_url)

collection = 'Duke-Breast-Cancer-MRI'
root_dir = Path('/mnt/IML-Proj/public_datasets/TCIA/Duke-Breast-Cancer-MRI/')

series_df = pd.DataFrame(
    data=client.get_json('getSeries', {'Collection': collection})
)
patient_df = pd.DataFrame(
    data=client.get_json('getPatientStudy', {'Collection': collection})
)

df = series_df.merge(patient_df)

file_df = pd.read_excel('Breast-Cancer-MRI-filepath_filename-mapping.xlsx')
file_df['SeriesInstanceUID'] = file_df['classic_path'].apply(lambda x: x.split('/')[3])
file_df['instance'] = file_df['original_path_and_filename'].apply(lambda x: x.split('/')[2])
file_df = file_df[['SeriesInstanceUID', 'instance']].drop_duplicates().reset_index(drop=True)

df = df.merge(file_df, on='SeriesInstanceUID', how='left')
df['instance'] = df['instance'].fillna('SEG')

for ii, row in df.iterrows():
    pid = row['PatientID']
    instance = row['instance']
    print(f'{pid}: {instance}')
    uid = row['SeriesInstanceUID']
    path = root_dir / (row['PatientID'] + f'/{instance}/{instance}.zip')
    if path.exists():
        print(f'exists: {path}')
        continue
    try:
        client.get_image(uid, path, True, True)
    except Exception as e:
        print(f'failed {e}')
