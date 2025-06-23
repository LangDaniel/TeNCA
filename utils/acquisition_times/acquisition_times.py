import pandas as pd

duke_df = pd.read_csv('./../../data/meta_info/duke_acquisition_times.csv') # path to retrieved DUKE acquisition times
mama_df = pd.read_excel('./../../data/MAMA-MIA/clinical_and_imaging_info.xlsx') # path to MAMA-MIA info file
mama_df = mama_df[['patient_id', 'acquisition_times']].rename(columns={'patient_id': 'identifier'})
duke_in_mama_ids = mama_df[mama_df['identifier'].str.startswith('DUKE')]['identifier'].values

# remove cases already included in mama mia
duke_df = duke_df[~duke_df['identifier'].isin(duke_in_mama_ids)]

df = pd.concat([mama_df, duke_df]).reset_index(drop=True)
# remove cases without acquisition times
df = df[~df['acquisition_times'].isna()]

df.to_csv('./../../data/meta_info/mama_duke_acquisition_times.csv', index=False)
