import pandas as pd
clin_path = 'datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]

gen_path = 'datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]

rad_path = 'datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
gen_df = gen_df[~gen_df.index.duplicated(keep='first')]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

df = pd.concat([clin_df[['ajcc_m']], gen_df, rad_df], axis=1, join='inner')
print(f"Overlap size: {df.shape[0]} patients")
