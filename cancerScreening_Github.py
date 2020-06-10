import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import random
# read in data
demo = pd.read_csv('fhg_demographics_2016_02_23_150948', sep='|', header=0)
demo = demo.drop_duplicates()
obs = pd.read_csv('fhg_observations_2016_02_23_143525', sep='|', header=0)
prob = pd.read_csv('fhg_problems_2016_02_23_011230', sep='|', header=0)
prob_cat_mapped = pd.read_csv('prob_cat_mapped.csv', header = 0)
CCS_cat_dx9 = prob_cat_mapped[prob_cat_mapped['CCS CATEGORY DESCRIPTION dx'].str.contains('can', na = False)]
CCS_cat_dx10 = prob_cat_mapped[prob_cat_mapped['CCS CATEGORY DESCRIPTION dx10'].str.contains('can', na = False)]
CCS_cat_px9 = prob_cat_mapped[prob_cat_mapped['CCS CATEGORY DESCRIPTION px'].str.contains('can', na = False)]
# Next consider the CCS_cat_dx9 only for colorectal, breast and cervical
CCS_cat_dx9 = CCS_cat_dx9.drop(['CCS CATEGORY DESCRIPTION dx10', 'CCS CATEGORY DESCRIPTION px', 'CCS CATEGORY dx10', 'CCS CATEGORY px', 'CCS Label_CPT', 'CCS_CPT'], axis = 1)
CCS_cat_dx9_breast = CCS_cat_dx9[CCS_cat_dx9['CCS CATEGORY DESCRIPTION dx'].str.contains('Breast cancr')]
# Only consider the data from 2010-2015 years
CCS_cat_dx9_breast['year'] = CCS_cat_dx9_breast['StartDate'].astype(str).str[0:4]
breast_sub = CCS_cat_dx9_breast[(CCS_cat_dx9_breast['year']>= '2010') & (CCS_cat_dx9_breast['year']<= '2015')]
patient_breast_yes_ID = breast_sub['PatientID'].unique()
breast_sub_demo = demo.merge(breast_sub, on = 'PatientID')
breast_sub_demo['age'] = (pd.to_datetime(breast_sub_demo['StartDate'])-pd.to_datetime(breast_sub_demo['DOB'])).astype('<m8[Y]')
breast_sub_demo.to_csv('breast_sub_demo_age.csv', header = True, index = False)
# select the patients has age [40, 69]
breast_sub_age = breast_sub_demo[(breast_sub_demo['age'] >= 40)& (breast_sub_demo['age']<=69)]
breast_sub_age_female = breast_sub_age[breast_sub_age['Gender']== 'Female']
breast_yes_ID = breast_sub_age_female['PatientID'].unique()
# Next find out the negative label
demo_female = demo[demo['Gender'] == 'Female']
demo_female['age2010'] = (pd.to_datetime('2010')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female['age2011'] = (pd.to_datetime('2011')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female['age2012'] = (pd.to_datetime('2012')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female['age2013'] = (pd.to_datetime('2013')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female['age2014'] = (pd.to_datetime('2014')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female['age2015'] = (pd.to_datetime('2015')-pd.to_datetime(demo_female['DOB'].str[0:4])).astype('<m8[Y]')
demo_female_age10 = demo_female[(demo_female['age2010']>= 40)& (demo_female['age2010']<= 69)]
demo_female_age11 = demo_female[(demo_female['age2011']>= 40)& (demo_female['age2011']<= 69)]
demo_female_age12 = demo_female[(demo_female['age2012']>= 40)& (demo_female['age2012']<= 69)]
demo_female_age13 = demo_female[(demo_female['age2013']>= 40)& (demo_female['age2013']<= 69)]
demo_female_age14 = demo_female[(demo_female['age2014']>= 40)& (demo_female['age2014']<= 69)]
demo_female_age15 = demo_female[(demo_female['age2015']>= 40)& (demo_female['age2015']<= 69)]
demo_female_age = demo_female_age10.append(demo_female_age11).append(demo_female_age12).append(demo_female_age13).append(demo_female_age14).append(demo_female_age15)
demo_female_age = demo_female_age.drop_duplicates()
breast_all_40_69_ID = demo_female_age['PatientID'].unique()
demo_female_age_No = demo_female_age[~demo_female_age['PatientID'].isin(breast_yes_ID)]
demo_female_age_obs_all = obs[obs['PatientID'].isin(breast_all_40_69_ID)]
demo_female_age_obs_Yes = obs[obs['PatientID'].isin(breast_yes_ID)]
breast_yes_ID_obs = demo_female_age_obs_Yes['PatientID'].unique()
n_breast_all = demo_female_age_obs_all['PatientID'].unique().shape
n_breast_yes = demo_female_age_obs_Yes['PatientID'].unique().shape
# Select out the patients with >=1 visits (visit time is different) for positive label
obs_sub = demo_female_age_obs_Yes[demo_female_age_obs_Yes['PatientID'].isin(breast_yes_ID_obs)]
obs_sub_visit = obs_sub[['PatientID','ResultDate']].drop_duplicates()
patient_ID_obs_unique = obs_sub_visit['PatientID'].unique()
n_obs_sub_unique = len(patient_ID_obs_unique)
count_visits = pd.DataFrame('', index=np.arange(n_obs_sub_unique), columns=['PatientID', 'Counts'])
n_row_obs_sub = len(obs_sub_visit)
for i in range(0, n_obs_sub_unique):
    print(i)
    ID = patient_ID_obs_unique[i]
    dat_i = obs_sub_visit[obs_sub_visit['PatientID'] == ID]
    count_visits.iloc[i, 0] = ID
    count_visits.iloc[i, 1] = len(dat_i)
# count_visits stores the visit times for each patient at different times
data_visits_valid = count_visits[count_visits['Counts'] >= 1]
patient_ID_valid = data_visits_valid['PatientID']
# Next find out the earlist dx date if have multiple visit times
obs_sub_visit = demo_female_age_obs_Yes[demo_female_age_obs_Yes['PatientID'].isin(patient_ID_valid)]
breast_sub_prob = breast_sub[breast_sub['PatientID'].isin(patient_ID_valid)]
# next find out the earliest dx time
patient_ID_prob_sim_unique = breast_sub_prob['PatientID'].unique()
early_date = pd.DataFrame('', index=np.arange(len(patient_ID_prob_sim_unique)), columns=['PatientID', 'Early date'])
for i in range(0, len(patient_ID_prob_sim_unique)):
    print(i)
    ID_sim = patient_ID_prob_sim_unique[i]
    dat_i_sim = breast_sub_prob[breast_sub_prob['PatientID'] == ID_sim]
    dat_i_sim = dat_i_sim.dropna(subset = ['StartDate'])
    early_date.iloc[i, 0] = ID_sim
    early_date.iloc[i, 1] = dat_i_sim.StartDate.min()
# Next find out the obs subset of visits before early date for each patient
column_name_obs = obs_sub_visit.columns.tolist()
obs_study = pd.DataFrame('', index = np.arange(len(obs_sub_visit)), columns = column_name_obs)
obs_study = obs_study.astype(str)
j = 0
for i in range(0, len(early_date)):
    print(i)
    ID_early = early_date['PatientID'][i]
    date_early = early_date['Early date'][i]
    dat_y = obs_sub_visit[obs_sub_visit['PatientID'] == ID_early]
    dat_y_sub = dat_y[dat_y['ResultDate']<= date_early]
    obs_study.iloc[j:j+len(dat_y_sub)] = dat_y_sub.values
    j = j+ len(dat_y_sub)
obs_study.replace('', np.nan, inplace=True)
obs_study1 = obs_study.dropna(how = 'all')
# Next find out the obs subset of visits AFTER early date for each patient
column_name_obs = obs_sub_visit.columns.tolist()
obs_study_after = pd.DataFrame('', index = np.arange(len(obs_sub_visit)), columns = column_name_obs)
obs_study_after = obs_study_after.astype(str)
j = 0
for i in range(0, len(early_date)):
    print(i)
    ID_early = early_date['PatientID'][i]
    date_early = early_date['Early date'][i]
    dat_y = obs_sub_visit[obs_sub_visit['PatientID'] == ID_early]
    dat_y_sub = dat_y[dat_y['ResultDate'] > date_early]
    obs_study_after.iloc[j:j+len(dat_y_sub)] = dat_y_sub.values
    j = j+ len(dat_y_sub)
obs_study_after.replace('', np.nan, inplace=True)
obs_study1_after = obs_study_after.dropna(how = 'all')
# ********************************************************************************************
# Next study the CVH values between patients have breast cancer screening and do not have
demo_female_age_obs_all = obs[obs['PatientID'].isin(breast_all_40_69_ID)]
demo_female_age_obs_Yes = obs_study1
demo_female_age_obs_Yes_after = obs_study1_after
demo_female_age_obs_No = demo_female_age_obs_all[~demo_female_age_obs_all['PatientID'].isin(breast_yes_ID)]
demo_female_age_obs_Yes = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ResultNumeric']!= -9.0]
demo_female_age_obs_Yes = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ResultString']!= '-9.0']
demo_female_age_obs_Yes_after = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ResultNumeric']!= -9.0]
demo_female_age_obs_Yes_after = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ResultString']!= '-9.0']
demo_female_age_obs_No = demo_female_age_obs_No[demo_female_age_obs_No['ResultNumeric']!= -9.0]
demo_female_age_obs_No = demo_female_age_obs_No[demo_female_age_obs_No['ResultString']!= '-9.0']
demo_female_age_obs_Yes.to_csv('demo_female_age_obs_Yes.csv', header = True, index = False)
demo_female_age_obs_Yes = pd.read_csv('demo_female_age_obs_Yes.csv', header = 0)
demo_female_age_obs_Yes_after.to_csv('demo_female_age_obs_Yes_after.csv', header = True, index = False)
demo_female_age_obs_Yes_after = pd.read_csv('demo_female_age_obs_Yes_after.csv', header = 0)
demo_female_age_obs_No.to_csv('demo_female_age_obs_No.csv', header = True, index = False)
demo_female_age_obs_No = pd.read_csv('demo_female_age_obs_No.csv', header = 0)
HGBA1C = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ObservationCategory'] == 'HGBA1C']
LDL = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ObservationCategory'] == 'LDL']
BMI = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ObservationCategory'] == 'BMI']
Smoking = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ObservationCategory'] == 'Smoking Status']
BP = demo_female_age_obs_Yes[demo_female_age_obs_Yes['ObservationCategory'] == 'Blood Pressure']
# Next replace the values to ideal, intermediate and poor
A1c = HGBA1C['ResultNumeric']
HGBA1C['Result'] = A1c
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c < 5.7], 'ideal')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][(A1c >= 5.7) & (A1c <= 6.4)], 'intermediate')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c > 6.4], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
diab_patient = pd.read_csv('diab_patient.csv', header =0) # 288045
diab_patient_date = diab_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates()
diab_patient_ID = pd.read_csv('diab_patient_ID.csv', header =0)
diab_patient_ID = diab_patient_ID['PatientID'].tolist()
idx1 = HGBA1C.index[HGBA1C['PatientID'].isin(diab_patient_ID)]
idx2 = HGBA1C.index[HGBA1C['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = HGBA1C.loc[idx]['PatientID'].iloc[ix]
    result = HGBA1C.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = diab_patient[diab_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
ldl = LDL['ResultNumeric']
LDL['Result'] = ldl
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl < 130], 'ideal')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][(ldl >= 130) & (ldl <= 159)], 'intermediate')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl > 159], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
chol_patient = pd.read_csv('chol_patient.csv', header =0)
chol_patient_date = chol_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates()
chol_patient_ID = pd.read_csv('chol_patient_ID.csv', header =0)
chol_patient_ID = chol_patient_ID['PatientID'].tolist()
idx1 = LDL.index[LDL['PatientID'].isin(chol_patient_ID)]
idx2 = LDL.index[LDL['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = LDL.loc[idx]['PatientID'].iloc[ix]
    result = LDL.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
print(j2)
print(j1)
bmi = BMI['ResultNumeric']
BMI['Result'] = bmi
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi < 25.0], 'ideal')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][(bmi >= 25.0) & (bmi < 30)], 'intermediate')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi >= 30.0], 'poor')
bp = BP['ResultString'].str.split('/')
bps = bp.str[0].astype(int)
bpd = bp.str[1].fillna(0).astype(int)
BP['Result'] = bps
BP['Result'] = BP['Result'].replace(BP['Result'][(bps < 120) & (bpd < 80)], 'ideal')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 120) & (bps <= 139)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bpd >= 80) & (bpd <= 89)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 140)|(bpd >= 90)], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
bp_patient = pd.read_csv('bp_patient.csv', header =0)
bp_patient_date = bp_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates()
bp_patient_ID = pd.read_csv('bp_patient_ID.csv', header =0)
bp_patient_ID = bp_patient_ID['PatientID'].tolist()
idx1 = BP.index[BP['PatientID'].isin(bp_patient_ID)]
idx2 = BP.index[BP['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = BP.loc[idx]['PatientID'].iloc[ix]
    result = BP.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
# Smoking has 53 different values and compolicated
smk = Smoking['ResultString'].str.strip().replace('.', '')
smk_no = ['No', 'never smoker', 'no', 'never', '0', 'never Never', 'NO', 'n', 'Negative', 'Non User', 'nonsmoker', 'never smoked']
smk_yes = ['Yes', 'yes', 'y', 'YES', 'Positive', 'current smoker', 'Tobacco User']
smk_else = [elem for elem in smk.unique().tolist() if not ((elem in smk_no) or (elem in smk_yes))]
Smoking['Result'] = smk
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_no)], 'ideal')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_else)], 'intermediate')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_yes)], 'poor')
data_converted_yes = HGBA1C.append(LDL).append(BMI).append(BP).append(Smoking)
# Since ther are multiple rows for smoking for one patient at the same day, need to
# keep one row by following the priority as: Yes -> No -> intermediate
Smoking2 = data_converted_yes[data_converted_yes['ObservationCategory']== 'Smoking Status']
smk_id = Smoking2['PatientID'].drop_duplicates() # 4793
for i in range(0, len(smk_id)):
    print(i)
    id_smk = smk_id.iloc[i]
    dat_smk = Smoking2[Smoking2['PatientID']== id_smk]
    idx1 = Smoking2.index[Smoking2['PatientID']== id_smk]
    smk_date = dat_smk['ResultDate'].drop_duplicates()
    for j in range(0, len(smk_date)):
        dat_smk_date = dat_smk[dat_smk['ResultDate'] == smk_date.iloc[j]]
        idx2 = Smoking2.index[Smoking2['ResultDate'] == smk_date.iloc[j]]
        idx = idx1.intersection(idx2)
        if(len(dat_smk_date) > 1):
            if((dat_smk_date['Result'] == 'ideal').any()):
                dat_smk_date['Result'] = 'ideal'
                data_converted_yes['Result'].loc[idx] = 'ideal'
            if((dat_smk_date['Result'] == 'poor').any()):
                dat_smk_date['Result'] = 'poor'
                data_converted_yes['Result'].loc[idx] = 'poor'
data_converted_yes = data_converted_yes.drop(['ResultString', 'ObservationName'], axis = 1)
data_converted_yes = data_converted_yes.drop_duplicates()
# Unitl here, the convertion for positive label was done
# Next convert the positive label after case
HGBA1C = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ObservationCategory'] == 'HGBA1C']
LDL = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ObservationCategory'] == 'LDL']
BMI = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ObservationCategory'] == 'BMI']
Smoking = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ObservationCategory'] == 'Smoking Status']
BP = demo_female_age_obs_Yes_after[demo_female_age_obs_Yes_after['ObservationCategory'] == 'Blood Pressure']
# Next replace the values to ideal, intermediate and poor
A1c = HGBA1C['ResultNumeric']
HGBA1C['Result'] = A1c
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c < 5.7], 'ideal')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][(A1c >= 5.7) & (A1c <= 6.4)], 'intermediate')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c > 6.4], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
diab_patient = pd.read_csv('diab_patient.csv', header =0) #
diab_patient_date = diab_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates()
diab_patient_ID = pd.read_csv('diab_patient_ID.csv', header =0)
diab_patient_ID = diab_patient_ID['PatientID'].tolist()
idx1 = HGBA1C.index[HGBA1C['PatientID'].isin(diab_patient_ID)]
idx2 = HGBA1C.index[HGBA1C['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = HGBA1C.loc[idx]['PatientID'].iloc[ix]
    result = HGBA1C.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = diab_patient[diab_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
ldl = LDL['ResultNumeric']
LDL['Result'] = ldl
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl < 130], 'ideal')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][(ldl >= 130) & (ldl <= 159)], 'intermediate')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl > 159], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
chol_patient = pd.read_csv('chol_patient.csv', header =0)
chol_patient_date = chol_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates() # 153766
chol_patient_ID = pd.read_csv('chol_patient_ID.csv', header =0)
chol_patient_ID = chol_patient_ID['PatientID'].tolist()
idx1 = LDL.index[LDL['PatientID'].isin(chol_patient_ID)]
idx2 = LDL.index[LDL['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = LDL.loc[idx]['PatientID'].iloc[ix]
    result = LDL.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
print(j2)
print(j1)
bmi = BMI['ResultNumeric']
BMI['Result'] = bmi
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi < 25.0], 'ideal')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][(bmi >= 25.0) & (bmi < 30)], 'intermediate')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi >= 30.0], 'poor')
bp = BP['ResultString'].str.split('/')
bps = bp.str[0].astype(int)
bpd = bp.str[1].fillna(0).astype(int)
BP['Result'] = bps
BP['Result'] = BP['Result'].replace(BP['Result'][(bps < 120) & (bpd < 80)], 'ideal')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 120) & (bps <= 139)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bpd >= 80) & (bpd <= 89)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 140)|(bpd >= 90)], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
bp_patient = pd.read_csv('bp_patient.csv', header =0)
bp_patient_date = bp_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates()
bp_patient_ID = pd.read_csv('bp_patient_ID.csv', header =0)
bp_patient_ID = bp_patient_ID['PatientID'].tolist()
idx1 = BP.index[BP['PatientID'].isin(bp_patient_ID)]
idx2 = BP.index[BP['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = BP.loc[idx]['PatientID'].iloc[ix]
    result = BP.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
# Smoking has 53 different values and compolicated
smk = Smoking['ResultString'].str.strip().replace('.', '')
smk_no = ['No', 'never smoker', 'no', 'never', '0', 'never Never', 'NO', 'n', 'Negative', 'Non User', 'nonsmoker', 'never smoked']
smk_yes = ['Yes', 'yes', 'y', 'YES', 'Positive', 'current smoker', 'Tobacco User']
smk_else = [elem for elem in smk.unique().tolist() if not ((elem in smk_no) or (elem in smk_yes))]
Smoking['Result'] = smk
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_no)], 'ideal')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_else)], 'intermediate')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_yes)], 'poor')
data_converted_yes_after = HGBA1C.append(LDL).append(BMI).append(BP).append(Smoking) # 15117
# Since ther are multiple rows for smoking for one patient at the same day, need to
# keep one row by following the priority as: Yes -> No -> intermediate
Smoking2 = data_converted_yes_after[data_converted_yes_after['ObservationCategory']== 'Smoking Status']
smk_id = Smoking2['PatientID'].drop_duplicates() # 4793
for i in range(0, len(smk_id)):
    print(i)
    id_smk = smk_id.iloc[i]
    dat_smk = Smoking2[Smoking2['PatientID']== id_smk]
    idx1 = Smoking2.index[Smoking2['PatientID']== id_smk]
    smk_date = dat_smk['ResultDate'].drop_duplicates()
    for j in range(0, len(smk_date)):
        dat_smk_date = dat_smk[dat_smk['ResultDate'] == smk_date.iloc[j]]
        idx2 = Smoking2.index[Smoking2['ResultDate'] == smk_date.iloc[j]]
        idx = idx1.intersection(idx2)
        if(len(dat_smk_date) > 1):
            if((dat_smk_date['Result'] == 'ideal').any()):
                dat_smk_date['Result'] = 'ideal'
                data_converted_yes_after['Result'].loc[idx] = 'ideal'
            if((dat_smk_date['Result'] == 'poor').any()):
                dat_smk_date['Result'] = 'poor'
                data_converted_yes_after['Result'].loc[idx] = 'poor'
data_converted_yes_after = data_converted_yes_after.drop(['ResultString', 'ObservationName'], axis = 1)
data_converted_yes_after = data_converted_yes_after.drop_duplicates()
# Until here the positive case after was done.
# Next study the negative case
ID_obs_No = set(demo_female_age_obs_No['PatientID'].unique())
import random
n_p = 8000
ID_obs_No_sub = random.sample(ID_obs_No, n_p)
demo_female_age_obs_No_sub = demo_female_age_obs_No[demo_female_age_obs_No['PatientID'].isin(ID_obs_No_sub)]
demo_female_age_obs_No_sub.to_csv('demo_female_age_obs_No_sub.csv', header = True, index = False)
demo_female_age_obs_No_sub = pd.read_csv('demo_female_age_obs_No_sub.csv', header = 0)
# Find out the CVH values for negative label
HGBA1C = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'HGBA1C']
LDL = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'LDL']
BMI = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'BMI']
Smoking = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'Smoking Status']
BP = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'Blood Pressure']
A1c = HGBA1C['ResultNumeric']
HGBA1C['Result'] = A1c
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c < 5.7], 'ideal')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][(A1c >= 5.7) & (A1c <= 6.4)], 'intermediate')
HGBA1C['Result'] = HGBA1C['Result'].replace(HGBA1C['Result'][A1c > 6.4], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
diab_patient = pd.read_csv('diab_patient.csv', header =0) # 288045
diab_patient_date = diab_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates() # 153766
diab_patient_ID = pd.read_csv('diab_patient_ID.csv', header =0)
diab_patient_ID = diab_patient_ID['PatientID'].tolist()
idx1 = HGBA1C.index[HGBA1C['PatientID'].isin(diab_patient_ID)]
idx2 = HGBA1C.index[HGBA1C['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = HGBA1C.loc[idx]['PatientID'].iloc[ix]
    result = HGBA1C.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = diab_patient[diab_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        HGBA1C.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
ldl = LDL['ResultNumeric']
LDL['Result'] = ldl
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl < 130], 'ideal')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][(ldl >= 130) & (ldl <= 159)], 'intermediate')
LDL['Result'] = LDL['Result'].replace(LDL['Result'][ldl > 159], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
chol_patient = pd.read_csv('chol_patient.csv', header =0) # 288045
chol_patient_date = chol_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates() # 153766
chol_patient_ID = pd.read_csv('chol_patient_ID.csv', header =0)
chol_patient_ID = chol_patient_ID['PatientID'].tolist()
idx1 = LDL.index[LDL['PatientID'].isin(chol_patient_ID)]
idx2 = LDL.index[LDL['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = LDL.loc[idx]['PatientID'].iloc[ix]
    result = LDL.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        LDL.loc[idx]['Result'].iloc[ix] = 'intermediate'
print(j2)
print(j1)
bmi = BMI['ResultNumeric']
BMI['Result'] = bmi
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi < 25.0], 'ideal')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][(bmi >= 25.0) & (bmi < 30)], 'intermediate')
BMI['Result'] = BMI['Result'].replace(BMI['Result'][bmi >= 30.0], 'poor')
bp = BP['ResultString'].str.split('/')
bps = bp.str[0].astype(int)
bpd = bp.str[1].fillna(0).astype(int)
BP['Result'] = bps
BP['Result'] = BP['Result'].replace(BP['Result'][(bps < 120) & (bpd < 80)], 'ideal')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 120) & (bps <= 139)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bpd >= 80) & (bpd <= 89)], 'intermediate')
BP['Result'] = BP['Result'].replace(BP['Result'][(bps >= 140)|(bpd >= 90)], 'poor')
# Next add the medication information
# Consider the Order date and Discontinue date
bp_patient = pd.read_csv('bp_patient.csv', header =0)
bp_patient_date = bp_patient[['PatientID', 'OrderDate', 'DiscontinueDate']].drop_duplicates() # 153766
bp_patient_ID = pd.read_csv('bp_patient_ID.csv', header =0)
bp_patient_ID = bp_patient_ID['PatientID'].tolist()
idx1 = BP.index[BP['PatientID'].isin(bp_patient_ID)]
idx2 = BP.index[BP['Result']=='ideal']
idx = idx1.intersection(idx2)
j1 = 0
j2 = 0
for ix in range(0, len(idx)):
    ID_ix = BP.loc[idx]['PatientID'].iloc[ix]
    result = BP.loc[idx].iloc[ix] # data for the CVH measures
    dat_med = chol_patient[chol_patient['PatientID'] == ID_ix] # data in the med
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (result['ResultDate'] < dat_med['DiscontinueDate']).any()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j1 = j1 + 1
    if((result['ResultDate'] > dat_med['OrderDate']).any() & (dat_med['DiscontinueDate'].isna()).all()):
        BP.loc[idx]['Result'].iloc[ix] = 'intermediate'
        j2 = j2 + 1
print(j2)
print(j1)
# Smoking has 53 different values and compolicated
smk = Smoking['ResultString'].str.strip().replace('.', '')
smk_no = ['No', 'never smoker', 'no', 'never', '0', 'never Never', 'NO', 'n', 'Negative', 'Non User', 'nonsmoker', 'never smoked']
smk_yes = ['Yes', 'yes', 'y', 'YES', 'Positive', 'current smoker', 'Tobacco User']
smk_else = [elem for elem in smk.unique().tolist() if not ((elem in smk_no) or (elem in smk_yes))]
Smoking['Result'] = smk
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_no)], 'ideal')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_else)], 'intermediate')
Smoking['Result'] = Smoking['Result'].replace(Smoking['Result'][smk.isin(smk_yes)], 'poor')
data_converted_no = HGBA1C.append(LDL).append(BMI).append(BP).append(Smoking)
# keep one row by following the priority as: Yes -> No -> intermediate
Smoking2 = data_converted_no[data_converted_no['ObservationCategory']== 'Smoking Status']
smk_id = Smoking2['PatientID'].drop_duplicates()
for i in range(0, len(smk_id)):
    print(i)
    id_smk = smk_id.iloc[i]
    dat_smk = Smoking2[Smoking2['PatientID']== id_smk]
    idx1 = Smoking2.index[Smoking2['PatientID']== id_smk]
    smk_date = dat_smk['ResultDate'].drop_duplicates()
    for j in range(0, len(smk_date)):
        dat_smk_date = dat_smk[dat_smk['ResultDate'] == smk_date.iloc[j]]
        idx2 = Smoking2.index[Smoking2['ResultDate'] == smk_date.iloc[j]]
        idx = idx1.intersection(idx2)
        if(len(dat_smk_date) > 1):
            if((dat_smk_date['Result'] == 'ideal').any()):
                dat_smk_date['Result'] = 'ideal'
                data_converted_no['Result'].loc[idx] = 'ideal'
            if((dat_smk_date['Result'] == 'poor').any()):
                dat_smk_date['Result'] = 'poor'
                data_converted_no['Result'].loc[idx] = 'poor'
data_converted_no = data_converted_no.drop(['ResultString', 'ObservationName'], axis = 1)
data_converted_no = data_converted_no.drop_duplicates() # 151547
# Unitl here, the convertion for positive label was done
# Next sort the dates of visits for each patients
df_yes = data_converted_yes
data_converted_yes = df_yes.sort_values(['PatientID', 'ResultDate'], ascending=[True, True])
df_yes_after = data_converted_yes_after
data_converted_yes_after = df_yes_after.sort_values(['PatientID', 'ResultDate'], ascending=[True, True])
df_no = data_converted_no
data_converted_no = df_no.sort_values(['PatientID', 'ResultDate'], ascending=[True, True])
data_converted_yes.to_csv('data_converted_yes_breast.csv', header = True, index = False)
data_converted_yes_after.to_csv('data_converted_yes_breast_after.csv', header = True, index = False)
data_converted_no.to_csv('data_converted_no_breast.csv', header = True, index = False)
data_converted_yes = pd.read_csv('data_converted_yes_breast.csv', header = 0)
data_converted_yes_after = pd.read_csv('data_converted_yes_breast_after.csv', header = 0)
data_converted_no = pd.read_csv('data_converted_no_breast.csv', header = 0)
# ******************************************************************************************
# Next check some statistics for the study datasets
data_converted_no['ObservationCategory'].value_counts()
counts_no = data_converted_no.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_no.iloc[0]/counts_no.iloc[0,:].sum() # BMI
counts_no.iloc[:,0].sum() # ideal
data_converted_yes['ObservationCategory'].value_counts()
counts_yes = data_converted_yes.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_yes.iloc[0]/counts_yes.iloc[0,:].sum() # BMI
counts_yes.iloc[:,0].sum() # ideal
# Next study if there is significantly different from positive patients and negative patients
# BMI, BP, A1C, LDL, Smoking
# str_ind = 'DEATH' # To indicate it is CHD(TYPE) or DEATH
# group3 = data_converted_yes # positive patients
# group2 = data_converted_no # negative patients
# n = 10 # n times re-sampling
# column_name = ['BMI_yes', 'BMI_no', 'BP_yes', 'BP_no', 'A1C_yes', 'A1C_yes', 'LDL_yes', 'LDL_no', 'Smoking_yes', 'Smoking_no']
# re_sampling_pert_df = pd.DataFrame('', index = np.arange(n), columns = column_name)
# for i in range(n):
#     print(i)
#     g3_sub = group3.sample(frac=0.7)
#     g3_chd_yes = g3_sub[g3_sub[str_ind] == 1]
#     g3_chd_pert = len(g3_chd_yes)/len(g3_sub)
#     re_sampling_pert_df.iloc[i, 0] = g3_chd_pert
#     g1_sub = group1.sample(frac=0.7)
#     g1_chd_yes = g1_sub[g1_sub[str_ind] == 1]
#     g1_chd_pert = len(g1_chd_yes)/len(g1_sub)
#     re_sampling_pert_df.iloc[i, 1] = g1_chd_pert
#     g2_sub = group2.sample(frac=0.7)
#     g2_chd_yes = g2_sub[g2_sub[str_ind] == 1]
#     g2_chd_pert = len(g2_chd_yes)/len(g2_sub)
#     re_sampling_pert_df.iloc[i, 2] = g2_chd_pert
#
# from scipy import stats
# re_sampling_pert_df[['Interactions', 'CVH', 'Treatment']].plot(kind='box')
# plt.title('Interactions, CVH, treatment and Death')
# # plt.xlabel('Categories')
# plt.ylabel('Percentages of Death')
# plt.savefig('figures/boxPlot_Interactions_CVH_Treatment_Death', dpi = 1200)
# Next apply the t- test to re_sampling_pert_df
# For BCS
re_sampling_pert_df_yes = [[13.8, 23.5, 62.6], [28.5, 48.2, 23.3], [11.9, 25.9, 62.2], [68.2, 17.6, 14.2], [72.8, 0.1, 27.1]]
re_sampling_pert_df_no = [[16.8, 23.8, 59.4], [28.8, 45.9, 25.3], [14.4, 32.4, 53.2], [69.4, 19.6, 11.0], [74.5, 0.2, 25.2]]
# For CECS
re_sampling_pert_df_yes = [[21.6, 32.0, 46.4], [36.3, 43.0, 20.8], [15.5, 26.2, 58.4], [68.8, 18.9, 12.3], [66.4, 0.0, 33.6]]
re_sampling_pert_df_no = [[17.4, 24.0, 58.7], [32.6, 44.3, 23.2], [18.1, 28.6, 53.2], [67.0, 20.6, 12.6], [72.9, 0.2, 26.9]]
# Fopr COCS
re_sampling_pert_df_yes = [[18.2, 27.3, 54.5], [23.2, 44.1, 32.7], [9.6, 18.7, 71.7], [81.4, 13.5, 5.1], [82.9, 0.2, 16.9]]
re_sampling_pert_df_no = [[17.7, 27.3, 55.0], [25.3, 47.0, 27.8], [15.1, 33.6, 51.3], [76.5, 14.7, 8.8], [73.2, 0.2, 26.6]]
from scipy.stats import ttest_ind
# from scipy.stats import f
from scipy.stats import chisquare
# Welch’s t-test
chi_square_BMI = chisquare(re_sampling_pert_df_yes[0], re_sampling_pert_df_no[0])
chi_square_BP = chisquare(re_sampling_pert_df_yes[1], re_sampling_pert_df_no[1])
chi_square_A1C = chisquare(re_sampling_pert_df_yes[2], re_sampling_pert_df_no[2])
chi_square_LDL = chisquare(re_sampling_pert_df_yes[3], re_sampling_pert_df_no[3])
chi_square_Smoking = chisquare(re_sampling_pert_df_yes[4], re_sampling_pert_df_no[4])
chi_square_BMI
chi_square_BP
chi_square_A1C
chi_square_LDL
chi_square_Smoking

# t_test_interaction_BP = ttest_ind(re_sampling_pert_df_yes[1], re_sampling_pert_df_no[1], equal_var= False)
# t_test_interaction_A1C = ttest_ind(re_sampling_pert_df_yes[2], re_sampling_pert_df_no[2], equal_var= False)
# t_test_interaction_LDL = ttest_ind(re_sampling_pert_df_yes[3], re_sampling_pert_df_no[3], equal_var= False)
# t_test_interaction_Smoking = ttest_ind(re_sampling_pert_df_yes[4], re_sampling_pert_df_no[4], equal_var= False)

t_test_interaction_TX = ttest_ind(re_sampling_pert_df.iloc[:, 0], re_sampling_pert_df.iloc[:, 2], equal_var= False)
t_test_interaction_TX
# Next plot this as a bar plot to display the difference between with or without screening
# study the same patients before and AFTER
ID_data_converted_yes = data_converted_yes['PatientID'].drop_duplicates().astype(int)
ID_data_converted_yes_after = set(data_converted_yes_after['PatientID'].tolist())
ID_data_converted_yes_before = set(ID_data_converted_yes.tolist())
ID_data_converted_yes_common = ID_data_converted_yes_before.intersection(ID_data_converted_yes_after)
data_converted_yes_before = data_converted_yes[data_converted_yes['PatientID'].isin(ID_data_converted_yes_common)]
data_converted_yes_after_common = data_converted_yes_after[data_converted_yes_after['PatientID'].isin(ID_data_converted_yes_common)]
data_converted_yes_before['ObservationCategory'].value_counts()
counts_yes_before = data_converted_yes_before.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_yes_before.iloc[0]/counts_yes_before.iloc[0,:].sum() # BMI
counts_yes_before.iloc[:,0].sum() # ideal
data_converted_yes_after_common['ObservationCategory'].value_counts()
counts_yes_after_common = data_converted_yes_after_common.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_yes_after_common.iloc[0]/counts_yes_after_common.iloc[0,:].sum() # BMI
counts_yes_after_common.iloc[:,0].sum() # ideal
# The bar plot for the comparison for ideal, intermediate and poor for patients with or without
# ************* The start of the plotiing for before and after cases
# 1. The IDEAL cases
barWidth = 0.5
bars1 = [11.5, 27.8, 11.1, 68.2, 71.8] # before
bars2 = [14.2, 30.4, 15.5, 69.8, 76.0] # after
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Ideal case of patients with BCS before and after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/ideal_patients_bcs_with_before_after_v2', dpi = 1200)
plt.show()
plt.gcf().clear()
# 2. The INTERMEDIATE cases
barWidth = 0.5
bars1 = [23.3, 49.1, 26.5, 17.6, 0.0] # have BCS
bars2 = [26.0, 47.9, 28.7, 20.9, 0.0] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Intermediate case of patients with BCS before after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/intermediate_patients_bcs_with_before_after_v2', dpi = 1200)
plt.show()
plt.gcf().clear()
# 1. The Poor cases
barWidth = 0.5
bars1 = [65.2, 23.1, 62.4, 14.3, 28.1] # have BCS
bars2 = [59.7, 21.7, 55.8, 9.4, 23.9] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Poor case of patients with BCS before after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/poor_patients_bcs_with_before_after_v2', dpi = 1200)
plt.show()
plt.gcf().clear()
# ************* The end of the plotiing for before and after cases
# *******Caculate the average numbers of visit times -- start
# positive case
data_converted_yes_all = data_converted_yes.append(data_converted_yes_after)
data_converted_yes_all = data_converted_yes_all.drop_duplicates()
demo_female_age_obs_Yes_all = demo_female_age_obs_Yes.append(demo_female_age_obs_Yes_after)
HGBA1C_yes = data_converted_yes_all[data_converted_yes_all['ObservationCategory'] == 'HGBA1C']
LDL_yes = data_converted_yes_all[data_converted_yes_all['ObservationCategory'] == 'LDL']
BMI_yes = data_converted_yes_all[data_converted_yes_all['ObservationCategory'] == 'BMI']
BP_yes = demo_female_age_obs_Yes_all[demo_female_age_obs_Yes_all['ObservationCategory'] == 'Blood Pressure']
BP_yes = BP_yes.drop(['ObservationName', 'ResultNumeric'], axis = 1) # 3676
BP_yes = BP_yes.drop_duplicates() # 2915
bp_yes = BP_yes['ResultString'].str.split('/')
bps_yes = bp_yes.str[0].astype(int)
bpd_yes = bp_yes.str[1].fillna(0).astype(int)
bps_yes.mean()
bps_yes.std()
bpd_yes.mean()
bpd_yes.std()
Smoking_yes = data_converted_yes_all[data_converted_yes_all['ObservationCategory'] == 'Smoking Status']
smk_yes = Smoking_yes[Smoking_yes['Result']== 'poor']
n_smk_yes = smk_yes['PatientID'].unique().shape
HGBA1C_yes['ResultNumeric'].mean()
HGBA1C_yes['ResultNumeric'].std()
LDL_yes['ResultNumeric'].mean()
LDL_yes['ResultNumeric'].std()
BMI_yes['ResultNumeric'].mean()
BMI_yes['ResultNumeric'].std()
# calculate the visit times
n_A1c_yes = len(HGBA1C_yes)
n_patient_A1c_yes = len(HGBA1C_yes['PatientID'].unique())
avg_times_A1c = n_A1c_yes/n_patient_A1c_yes
n_LDL_yes = len(LDL_yes)
n_patient_LDL_yes = len(LDL_yes['PatientID'].unique())
avg_times_LDL = n_LDL_yes/n_patient_LDL_yes
n_BMI_yes = len(BMI_yes)
n_patient_BMI_yes = len(BMI_yes['PatientID'].unique())
avg_times_BMI = n_BMI_yes/n_patient_BMI_yes
n_BP_yes = len(BP_yes)
n_patient_BP_yes = len(BP_yes['PatientID'].unique())
avg_times_BP = n_BP_yes/n_patient_BP_yes
n_Smoking_yes = len(Smoking_yes)
n_patient_Smoking_yes = len(Smoking_yes['PatientID'].unique())
avg_times_Smoking = n_Smoking_yes/n_patient_Smoking_yes
# Negative case
HGBA1C_no = data_converted_no[data_converted_no['ObservationCategory'] == 'HGBA1C']
LDL_no = data_converted_no[data_converted_no['ObservationCategory'] == 'LDL']
BMI_no = data_converted_no[data_converted_no['ObservationCategory'] == 'BMI']
BP_no = demo_female_age_obs_No_sub[demo_female_age_obs_No_sub['ObservationCategory'] == 'Blood Pressure']
BP_no = BP_no.drop(['ObservationName', 'ResultNumeric'], axis = 1) # 3676
BP_no = BP_no.drop_duplicates() # 2915
bp_no = BP_no['ResultString'].str.split('/')
bps_no = bp_no.str[0].astype(int)
bpd_no = bp_no.str[1].fillna(0).astype(int)
bps_no.mean()
bps_no.std()
bpd_no.mean()
bpd_no.std()
Smoking_no = data_converted_no[data_converted_no['ObservationCategory'] == 'Smoking Status']
smk_no = Smoking_no[Smoking_no['Result']== 'poor']
n_smk_no = smk_no['PatientID'].unique().shape
HGBA1C_no['ResultNumeric'].mean()
HGBA1C_no['ResultNumeric'].std()
LDL_no['ResultNumeric'].mean()
LDL_no['ResultNumeric'].std()
BMI_no['ResultNumeric'].mean()
BMI_no['ResultNumeric'].std()
# calculate the visit times
n_A1c_no = len(HGBA1C_no)
n_patient_A1c_no = len(HGBA1C_no['PatientID'].unique())
avg_times_A1c_no = n_A1c_no/n_patient_A1c_no
n_LDL_no = len(LDL_no)
n_patient_LDL_no = len(LDL_no['PatientID'].unique())
avg_times_LDL_no = n_LDL_no/n_patient_LDL_no
n_BMI_no = len(BMI_no)
n_patient_BMI_no = len(BMI_no['PatientID'].unique())
avg_times_BMI_no = n_BMI_no/n_patient_BMI_no
n_BP_no = len(BP_no)
n_patient_BP_no = len(BP_no['PatientID'].unique())
avg_times_BP_no = n_BP_no/n_patient_BP_no
n_Smoking_no = len(Smoking_no)
n_patient_Smoking_no = len(Smoking_no['PatientID'].unique())
avg_times_Smoking_no = n_Smoking_no/n_patient_Smoking_no
# Next plot the visit times for the positive case and negative case
barWidth = 0.5
bars1 = [9.81, 11.23, 4.31, 3.03, 7.69] # have BCS
bars2 = [6.72, 8.08, 3.40, 2.54, 5.35] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='BCS')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='No BCS')#, edgecolor='black')
plt.title('Average visit times of patients with/out BCS')
# plt.xlabel('CVH categories')
plt.ylabel('Average number of visits')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/Avg_visit_times_bcs_with_without_v3', dpi = 1200)
plt.show()
plt.gcf().clear()
# *******Caculate the average numbers of visit times --end
# The bar plot for the comparison for ideal, intermediate and poor for patients with or without
# 1. The IDEAL cases
barWidth = 0.5
bars1 = [13.8, 28.5, 11.9, 68.2, 72.8] # have BCS
bars2 = [16.8, 28.8, 14.4, 69.4, 74.5] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='BCS')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='No BCS')#, edgecolor='black')
plt.title('Ideal case of patients with BCS and without BCS')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/ideal_patients_bcs_with_without_v3', dpi = 1200)
plt.show()
plt.gcf().clear()
# 2. The INTERMEDIATE cases
barWidth = 0.5
bars1 = [23.5, 48.2, 25.9, 17.6, 0.1] # have BCS
bars2 = [23.8, 45.9, 32.4, 19.6, 0.2] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='BCS')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='No BCS')#, edgecolor='black')
plt.title('Intermediate case of patients with BCS and without BCS')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/intermediate_patients_bcs_with_without_v3', dpi = 1200)
plt.show()
plt.gcf().clear()
# 1. The IDEAL cases
barWidth = 0.5
bars1 = [62.6, 23.3, 62.2, 14.2, 27.1] # have BCS
bars2 = [59.4, 25.3, 53.2, 11.0, 25.2] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='BCS')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='No BCS')#, edgecolor='black')
plt.title('Poor case of patients with BCS and without BCS')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/poor_patients_bcs_with_without_v3', dpi = 1200)
plt.show()
plt.gcf().clear()
# ***** The end of the plotting of avg visit times
# Next investigate the AGE information for the positive case
ID_data_converted_yes = data_converted_yes['PatientID'] # 565 unique patients
ID_data_converted_yes = ID_data_converted_yes.drop_duplicates().to_frame()
data_converted_yes_age = ID_data_converted_yes.merge(breast_sub_demo, on = 'PatientID')
data_converted_yes_age_sub = data_converted_yes_age[['PatientID', 'age']] # 1197
data_converted_yes_age_sub = data_converted_yes_age_sub.drop_duplicates()
df_yes = data_converted_yes_age_sub
data_converted_yes_age_sub = df_yes.sort_values(['PatientID', 'age'], ascending=[True, True])
data_converted_yes_age_sub = data_converted_yes_age_sub.drop_duplicates(subset='PatientID', keep='first')
# next plot the age
age = data_converted_yes_age_sub['age']
# Divided the age to 3 groups 21-35, 36-50 and 51-64 to bar plot for the positive case
age1 = age
barWidth = 1.0
age1[(age >=40) & (age <= 49)] = 2 # group 21-35 as 2
age1[(age > 49) & (age <= 59)] = 1 # group 35-50 as 1
age1[(age > 59) & (age <= 69)] = 0 # group 50-64 as 0
age1.value_counts()
count_age = [172, 282, 270]
type_age = ['40-49', '50-59', '60-69']
plt.bar(type_age, count_age, width =0.5 * barWidth, color = ('green', 'violet', 'orange'))
plt.title('Counts of three age groups for patients with BCS')
# plt.xlabel('Groups')
plt.ylabel('Counts of age')
# Create the percentages for each bar
# plt.legend()
pert_0 = '{:.1%}'.format(count_age[0]/(sum(count_age)))
pert_1 = '{:.1%}'.format(count_age[1]/(sum(count_age)))
pert_2 = '{:.1%}'.format(count_age[2]/(sum(count_age)))
label = [pert_0, pert_1, pert_2]
# Text on the top of each barplot
r1 = [0,1,2]
plt.text(x = r1[0]-0.1 , y = count_age[0], s = label[0], size = 10)
plt.text(x = r1[1]-0.1 , y = count_age[1], s = label[1], size = 10)
plt.text(x = r1[2]-0.1 , y = count_age[2], s = label[2], size = 10)
plt.savefig('figures/Age_grouped_counts_three_BCS_v2', dpi = 1200)
plt.show()
plt.gcf().clear()
# Divided the age to 3 groups 21-35, 36-50 and 51-64 to bar plot for the positive case and negiative case
data_converted_no_demo = data_converted_no.merge(demo, on = 'PatientID')
data_converted_no_demo = data_converted_no_demo.drop_duplicates()
data_converted_no_demo['age'] = (pd.to_datetime(data_converted_no_demo['ResultDate'])-pd.to_datetime(data_converted_no_demo['DOB'])).astype('<m8[Y]')
data_converted_no_age = data_converted_no_demo[['PatientID', 'age']]
data_converted_no_age = data_converted_no_age.drop_duplicates()
data_converted_no_age = data_converted_no_age.sort_values(['PatientID', 'age'], ascending =[True, True])
data_converted_no_age = data_converted_no_age.drop_duplicates(subset = 'PatientID', keep = 'last')
age_no = data_converted_no_age['age']
age1 = age_no
barWidth = 1.0
age1[(age_no >=40) & (age_no <= 49)] = 2 # group 21-35 as 2
age1[(age_no > 49) & (age_no <= 59)] = 1 # group 35-50 as 1
age1[(age_no > 59) & (age_no <= 69)] = 0 # group 50-64 as 0
age1.value_counts()
count_age_no = [2184, 2502, 1999]
count_age = [2184, 2502, 1999]
type_age = ['40-49', '50-59', '60-69']
plt.bar(type_age, count_age, width =0.5 * barWidth, color = ('green', 'violet', 'orange'))
plt.title('Counts of three age groups for patients without BCS')
# plt.xlabel('Groups')
plt.ylabel('Counts of age')
# Create the percentages for each bar
# plt.legend()
pert_0 = '{:.1%}'.format(count_age[0]/(sum(count_age)))
pert_1 = '{:.1%}'.format(count_age[1]/(sum(count_age)))
pert_2 = '{:.1%}'.format(count_age[2]/(sum(count_age)))
label = [pert_0, pert_1, pert_2]
# Text on the top of each barplot
r1 = [0,1,2]
plt.text(x = r1[0]-0.1 , y = count_age[0], s = label[0], size = 10)
plt.text(x = r1[1]-0.1 , y = count_age[1], s = label[1], size = 10)
plt.text(x = r1[2]-0.1 , y = count_age[2], s = label[2], size = 10)
plt.savefig('figures/Age_grouped_counts_three_BCS_no', dpi = 1200)
plt.show()
plt.gcf().clear()
# ***** The end of the age plots for both cases (not done yet)
# Next study the race information for the positive cases
# ID_data_converted_no = data_converted_no['PatientID'].to_frame()
# ID_data_converted_no = ID_data_converted_no.drop_duplicates()
# demo_yes_no = ID_data_converted_no.merge(demo, on = 'PatientID')
demo_yes_no = ID_data_converted_yes.merge(demo, on = 'PatientID')
race_yes_no = demo_yes_no[['PatientID', 'Race']].drop_duplicates()
race = race_yes_no['Race'].str.strip().replace('.', '')
race_yes_no_nan = race_yes_no[race_yes_no['Race'].isna()]
race_yes_no = race_yes_no[~race_yes_no['Race'].isna()]
race_white = ['White', 'White/Caucasian', 'White-Italian', 'C-CAUCASIAN']
race_unknown = ['Patient Declined', 'X-PATIENT DECLINED', 'Refuse to Report/ Unreported']
race_non_white = [elem for elem in race.unique().tolist() if not ((elem in race_white) or (elem in race_unknown))]
race_yes_no_white = race_yes_no[race.isin(race_white)].drop_duplicates()
n_race_white = len(race_yes_no_white)
race_yes_no_unknown = race_yes_no[race.isin(race_unknown)].drop_duplicates()
n_race_yes_no_unknown = len(race_yes_no_unknown)+ len(race_yes_no_nan)
race_yes_no_nonwhite = race_yes_no[race.isin(race_non_white)].drop_duplicates()
n_race_nonwhite = len(race_yes_no_nonwhite)
# next plot the race information
count_race = [n_race_nonwhite, n_race_white, n_race_yes_no_unknown]
type_race = ['Non-White', 'White', 'Unknown']
plt.bar(type_race, count_race, width = 0.5* barWidth, color = ('green', 'blue', 'orange'))
plt.title('Counts of three race categories with BCS')
# plt.xlabel('Categories')
plt.ylabel('Counts of race')
# Create the percentages for each bar
# plt.legend()
pert_NonW = '{:.1%}'.format(count_race[0]/(sum(count_race)))
pert_W = '{:.1%}'.format(count_race[1]/(sum(count_race)))
pert_U = '{:.1%}'.format(count_race[2]/(sum(count_race)))
label = [pert_NonW, pert_W, pert_U]
# Text on the top of each barplot
r1 = [0,1,2]
plt.text(x = r1[0]-0.1 , y = count_race[0], s = label[0], size = 10)
plt.text(x = r1[1]-0.1 , y = count_race[1], s = label[1], size = 10)
plt.text(x = r1[2]-0.1 , y = count_race[2], s = label[2], size = 10)
plt.savefig('figures/Race_counts_three_BCS_v2', dpi = 1200)
plt.show()
plt.gcf().clear()
# ***** the end of the age and race plots
# Study the negative label case for the patients CVH changes
data_converted_no_both = data_converted_no
column_no = data_converted_no_both.columns.tolist()
id_both = data_converted_no_both['PatientID'].unique()
n_id_both = len(id_both)
data_no_before = pd.DataFrame('', index = np.arange(len(data_converted_no_both)), columns = column_no)
data_no_after = pd.DataFrame('', index = np.arange(len(data_converted_no_both)), columns = column_no)
m = 0
n = 0
for i in range(0, n_id_both):
    print(i)
    id_no = id_both[i]
    dat_x_no = data_converted_no_both[data_converted_no_both['PatientID'] == id_no]
    n_date = len(dat_x_no['ResultDate'].unique())
    dat_no_before_i = dat_x_no[dat_x_no['ResultDate']<= dat_x_no['ResultDate'].iloc[n_date // 2]]
    data_no_before.iloc[m: m+len(dat_no_before_i)] = dat_no_before_i.values
    m = m + len(dat_no_before_i)
    dat_no_after_i = dat_x_no[dat_x_no['ResultDate']> dat_x_no['ResultDate'].iloc[n_date // 2]]
    data_no_after.iloc[n: n+len(dat_no_after_i)] = dat_no_after_i.values
    n = n + len(dat_no_after_i)
data_no_before = data_no_before.drop_duplicates()
data_no_before = data_no_before[:-1]
data_no_after = data_no_after.drop_duplicates()
data_no_after = data_no_after[:-1]
data_no_before = data_no_before.to_csv('data_no_before.csv', header = True, index = False)
data_no_after = data_no_after.to_csv('data_no_after.csv', header = True, index = False)
# Now can read in these files directly
data_no_before = pd.read_csv('data_no_before.csv', header = 0)
data_no_after = pd.read_csv('data_no_after.csv', header = 0)
# Next study the patients CVH changes from the before and after datasets
# study the SAME patients before and AFTER for the negative case
# ID_data_converted_yes = data_converted_yes['PatientID'].astype(int) # 565 unique patients
ID_data_converted_no_after = set(data_no_after['PatientID'].tolist())
ID_data_converted_no_before = set(data_no_before['PatientID'].tolist())
ID_data_converted_no_common = ID_data_converted_no_before.intersection(ID_data_converted_no_after)
data_converted_no_before = data_no_before[data_no_before['PatientID'].isin(ID_data_converted_no_common)]
data_converted_no_after = data_no_after[data_no_after['PatientID'].isin(ID_data_converted_no_common)]
data_converted_no_before['ObservationCategory'].value_counts()
counts_no_before = data_converted_no_before.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_no_before.iloc[0]/counts_no_before.iloc[0,:].sum() # BMI
counts_no_before.iloc[:,0].sum() # ideal
data_converted_no_after['ObservationCategory'].value_counts()
counts_no_after = data_converted_no_after.groupby(['ObservationCategory', 'Result']).size().unstack()
counts_no_after.iloc[0]/counts_no_after.iloc[0,:].sum() # BMI
counts_no_after.iloc[:,0].sum() # ideal
# The bar plot for the comparison for ideal, intermediate and poor for patients with or without
# ************* The start of the plotiing for before and after cases for negative label
# 1. The IDEAL cases
barWidth = 0.5
bars1 = [16.4, 28.1, 16.4, 67.8, 75.4] # before
bars2 = [16.7, 28.9, 13.2, 70.7, 73.8] # after
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Ideal case of patients without BCS before and after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/ideal_patients_bcs_without_before_after', dpi = 1200)
plt.show()
plt.gcf().clear()
# 2. The INTERMEDIATE cases
barWidth = 0.5
bars1 = [25.0, 44.2, 31.5, 20.5, 0.2] # have BCS
bars2 = [23.1, 46.6, 32.9, 18.7, 0.1] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Intermediate case of patients without BCS before after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/intermediate_patients_bcs_without_before_after', dpi = 1200)
plt.show()
plt.gcf().clear()
# 1. The Poor cases
barWidth = 0.5
bars1 = [58.6, 27.7, 52.1, 11.7, 24.3] # have BCS
bars2 = [60.2, 24.4, 53.8, 10.5, 26.1] # do not have BCS
bars4 = bars1 + bars2
# The X position of bars
r1 = [0.5, 2, 3.5, 5, 6.5]
r2 = [1, 2.5, 4, 5.5, 7]
r4 = r1 + r2
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = 'green', label='Before')#, edgecolor='black')
plt.bar(r2, bars2, width = barWidth, color = 'orange', label='After')#, edgecolor='black')
plt.title('Poor case of patients without BCS before after')
# plt.xlabel('CVH categories')
plt.ylabel('Percentage %')
# Create legend
plt.legend()
# Text below each barplot
plt.xticks([r+0.75 for r in [0, 1.5, 3, 4.5, 6]], ['BMI', 'BP', 'A1C', 'LDL', 'Smoking'])
# Create labels
pert_0_N = bars1[0]
pert_1_N = bars1[1]
pert_2_N = bars1[2]
pert_3_N = bars1[3]
pert_4_N = bars1[4]
pert_0_Y = bars2[0]
pert_1_Y = bars2[1]
pert_2_Y = bars2[2]
pert_3_Y = bars2[3]
pert_4_Y = bars2[4]
label = [pert_0_N, pert_1_N, pert_2_N, pert_3_N, pert_4_N, pert_0_Y, pert_1_Y, pert_2_Y, pert_3_Y, pert_4_Y]
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2, y = bars4[i], s = label[i], size = 8.5) # y = bars4[i]-bars4[i]/2
plt.savefig('figures/poor_patients_bcs_without_before_after', dpi = 1200)
plt.show()
plt.gcf().clear()
# ************* The end of the plotiing for before and after cases for negative label
# ******************************************************************************************
# Next time they can be read in directly
data_converted_yes = pd.read_csv('data_converted_yes_breast.csv', header = 0)
data_converted_no = pd.read_csv('data_converted_no_breast.csv', header = 0)
#************************************************************************************
# Here is the start of SLTM
# select out the 10% subset of no case randomly
# And then run ten times to get the values
nN = 10
Accuracy = pd.DataFrame('', index = np.arange(nN), columns=['Accuracy'])
shape_roc = pd.DataFrame('', index = np.arange(nN), columns=['Length'])
FPR_TPR_T_ROC = pd.DataFrame('', index = np.arange(nN*200), columns=['fpr_keras', 'tpr_keras', 'thresholds_keras'])
AUCROC = pd.DataFrame('', index = np.arange(nN), columns=['AUCROC'])
k = 0
j = 0
for i in range(0, nN):
    data_converted_no_all = data_converted_no
    n_p_no = 800
    ID_no = set(data_converted_no_all['PatientID'].unique())
    ID_no_selected = random.sample(ID_no, n_p_no)
    data_converted_no = data_converted_no_all[data_converted_no_all['PatientID'].isin(ID_no_selected)]
    # Next combine the result name and result values columns
    data_converted_yes['ObsResult'] = data_converted_yes['ObservationCategory'].astype(str) + data_converted_yes['Result'].astype(str)
    data_converted_no['ObsResult'] = data_converted_no['ObservationCategory'].astype(str) + data_converted_no['Result'].astype(str)
    data_converted_yes_sub = data_converted_yes[['PatientID', 'ObsResult', 'ResultDate']]
    data_converted_no_sub = data_converted_no[['PatientID', 'ObsResult', 'ResultDate']]
    data_converted_sub1 = data_converted_yes_sub.append(data_converted_no_sub)
    data_converted_sub = data_converted_sub1
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statusideal', 'SmokingStatusideal')
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statusintermediate', 'SmokingStatusintermediate')
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Smoking Statuspoor', 'SmokingStatuspoor')
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressureideal', 'BloodPressureideal')
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressureintermediate', 'BloodPressureintermediate')
    data_converted_sub['ObsResult'] = data_converted_sub1['ObsResult'].replace('Blood Pressurepoor', 'BloodPressurepoor')
    data_converted_sub['ObsResult'] = data_converted_sub['ObsResult'].str.lower()
    # Next apply the LSTM model to the above data
    data_converted_sub_v1 = data_converted_sub
    # Next apply word2vec to the 'Code' column of data_converted_sub_v1
    chd = data_converted_sub_v1
    chd['ObsResult'] = chd['ObsResult'].astype(str)
    patients_v3 = chd.groupby('PatientID')['ObsResult'].apply(list)
    patients_v2 = list(patients_v3)
    patients_all_v1 = patients_v2
    from gensim.models import Word2Vec
    nv = 32
    model = Word2Vec(patients_all_v1, min_count = 1, size = nv)
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    # print(words)
    # model.wv.save_word2vec_format("vectors_15.csv", binary=False) # save the embeddings
    # # The direct output from embedding (directly read in)
    # vectors = pd.read_csv("vectors_15.csv", header = None)
    vector_all = pd.DataFrame(0, index= np.arange(len(words)), columns = list(range(nv+1)))
    i = 0
    for word in words:
        print(i)
        vector_all.iloc[i, 0] = word
        vector_all.iloc[i, 1:] = model[word]
        i = i + 1
    vector_all = vector_all.rename(columns={0: 'ObsResult'})
    vector_all['ObsResult'] = vector_all['ObsResult'].astype(str)
    data_converted_sub_v1['ObsResult'] = data_converted_sub_v1['ObsResult'].astype(str)
    data_converted_map = vector_all.merge(data_converted_sub_v1, on = 'ObsResult')
    data_converted_map = data_converted_map.sort_values(['PatientID', 'ResultDate'], ascending=[True, True])
    max_date = data_converted_map.groupby('PatientID', as_index=False).last()[['PatientID', 'ResultDate']]
    max_date = max_date.rename(columns = {'ResultDate':'max_date'})
    data_converted_map1 = data_converted_map.merge(max_date, on = 'PatientID')
    data_converted_map1['days'] = (pd.to_datetime(data_converted_map1['ResultDate'])-pd.to_datetime(data_converted_map1['max_date'])).astype('<m8[D]')
    data_converted_map1 = data_converted_map1[data_converted_map1['days'].notna()]
    # Convert the rows to columns for one patient
    chd = data_converted_map1.drop(columns = ['ResultDate', 'max_date', 'ObsResult'])
    id_chd = chd['PatientID'].unique()
    n_row = len(id_chd)
    patient_ID = pd.DataFrame('', index=np.arange(n_row), columns=['PatientID'])
    patients_row_v2 = []
    patients = []
    for i in range(0, n_row):
        x = id_chd[i]
        # print(i)
        dat_x = chd[chd['PatientID']== x] # find out the subset for each id
        n_row_x = len(dat_x)
        y = dat_x.drop(columns = 'PatientID')
        y = y.values.tolist()
        patients.append(y)
    patients_all_v2 = patients
    # from keras.utils.np_utils import to_categorical
    # fix random seed for reproducibility
    # numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    word = data_converted_sub['ObsResult'].unique()
    top_words = len(word)+1
    y_yes = np.zeros(len(data_converted_yes_sub['PatientID'].unique()))+1
    y_no = np.zeros(len(data_converted_no_sub['PatientID'].unique()))
    y = np.append(y_yes, y_no)
    # y = to_categorical(y)
    features1 = np.array(patients_all_v2)
    features = features1
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.20)
    # truncate and pad input sequences
    top_words = len(word) + 1
    nM = int(chd['PatientID'].value_counts().max())
    max_review_length = nM
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32 + 1
    #*****************************************************************************
    # By LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(max_review_length, embedding_vecor_length)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss='mean_squared_error'
    print(model.summary())
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=10), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    model.fit(X_train, y_train, epochs=100, callbacks = callbacks, verbose=1,batch_size=64, validation_data=(X_test, y_test))
    # If use Embedding but not by Word2Vec() without delta t info - end
    #********************************************************************************
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    Accuracy.iloc[k] = scores[1]
    # Next store the ROC values for the later plot
    y_pred_keras = model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    nL = len(fpr_keras)
    FPR_TPR_T_ROC.iloc[j:(j+nL), 0] = fpr_keras
    FPR_TPR_T_ROC.iloc[j:(j+nL), 1] = tpr_keras
    FPR_TPR_T_ROC.iloc[j:(j+nL), 2] = thresholds_keras
    j = j+nL
    AUCROC.iloc[k] = auc_keras
    shape_roc.iloc[k] = len(fpr_keras)
    k = k + 1

Accuracy.to_csv('Accuracy_breast_10_times_v3.csv', header = True, index = False)
FPR_TPR_T_ROC.to_csv('FPR_TPR_T_ROC_breast_10_times_v3.csv', header = True, index = False)
AUCROC.to_csv('AUCROC_breast_10_times_v3.csv', header = True, index = False)
shape_roc.to_csv('shape_roc_breast_10_times_v3.csv', header = True, index = False)
# Here is the end of SLTM
#************************************************************************************
# plot the Accuracy
# Accuracy.plot(kind = 'bar')
# plt.savefig('figures/Accuracy_CVH_LSTM_Breast_10_times_v3', dpi = 1200)
# AUCROC.plot(kind = 'bar')
# plt.savefig('figures/AUCROC_CVH_LSTM_Breast_10_times_v3', dpi = 1200)
auc = AUCROC['AUCROC'].tolist()
shape = shape_roc['Length'].tolist()
n_min = min(shape)
n1 = shape[0]
fpr_keras1 = FPR_TPR_T_ROC.iloc[0:n1, 0]
tpr_keras1 = FPR_TPR_T_ROC.iloc[0:n1, 1]
n2 = shape[1]
fpr_keras2 = FPR_TPR_T_ROC.iloc[(n1+1):(n1+n2), 0]
tpr_keras2 = FPR_TPR_T_ROC.iloc[(n1+1):(n1+n2), 1]
n3 = shape[2]
fpr_keras3 = FPR_TPR_T_ROC.iloc[(n1+n2+1):(n1+n2+n3), 0]
tpr_keras3 = FPR_TPR_T_ROC.iloc[(n1+n2+1):(n1+n2+n3), 1]
n4 = shape[3]
fpr_keras4 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+1):(n1+n2+n3+n4), 0]
tpr_keras4 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+1):(n1+n2+n3+n4), 1]
n5 = shape[4]
fpr_keras5 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+1):(n1+n2+n3+n4+n5), 0]
tpr_keras5 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+1):(n1+n2+n3+n4+n5), 1]
n6 = shape[5]
fpr_keras6 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+1):(n1+n2+n3+n4+n5+n6), 0]
tpr_keras6 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+1):(n1+n2+n3+n4+n5+n6), 1]
n7 = shape[6]
fpr_keras7 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+1):(n1+n2+n3+n4+n5+n6+n7), 0]
tpr_keras7 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+1):(n1+n2+n3+n4+n5+n6+n7), 1]
n8 = shape[7]
fpr_keras8 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+1):(n1+n2+n3+n4+n5+n6+n7+n8), 0]
tpr_keras8 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+1):(n1+n2+n3+n4+n5+n6+n7+n8), 1]
n9 = shape[8]
fpr_keras9 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+n8+1):(n1+n2+n3+n4+n5+n6+n7+n8+n9), 0]
tpr_keras9 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+n8+1):(n1+n2+n3+n4+n5+n6+n7+n8+n9), 1]
n10 = shape[9]
fpr_keras10 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+n8+n9+1):(n1+n2+n3+n4+n5+n6+n7+n8+n9+n10), 0]
tpr_keras10 = FPR_TPR_T_ROC.iloc[(n1+n2+n3+n4+n5+n6+n7+n8+n9+1):(n1+n2+n3+n4+n5+n6+n7+n8+n9+n10), 1]
# Next plot the 10 times roc curves for the LSTM
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.005,1.005])
plt.ylim([-0.005,1.005])
plt.plot(fpr_keras1, tpr_keras1, label='LSTM1 (AUC = {:.2f})'.format(auc[0]))
plt.plot(fpr_keras2, tpr_keras2, label='LSTM2 (AUC = {:.2f})'.format(auc[1]))
plt.plot(fpr_keras3, tpr_keras3, label='LSTM3 (AUC = {:.2f})'.format(auc[2]))
plt.plot(fpr_keras4, tpr_keras4, label='LSTM4 (AUC = {:.2f})'.format(auc[3]))
plt.plot(fpr_keras5, tpr_keras5, label='LSTM5 (AUC = {:.2f})'.format(auc[4]))
plt.plot(fpr_keras6, tpr_keras6, label='LSTM6 (AUC = {:.2f})'.format(auc[5]))
plt.plot(fpr_keras7, tpr_keras7, label='LSTM7 (AUC = {:.2f})'.format(auc[6]))
plt.plot(fpr_keras8, tpr_keras8, label='LSTM8 (AUC = {:.2f})'.format(auc[7]))
plt.plot(fpr_keras9, tpr_keras9, label='LSTM9 (AUC = {:.2f})'.format(auc[8]))
plt.plot(fpr_keras10, tpr_keras10, label='LSTM10 (AUC = {:.2f})'.format(auc[9]))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc='best')
plt.savefig('figures/roc_CVH_LSTM_Breast_10_times_onePlot_v3_v3', dpi = 1200)
plt.show()
