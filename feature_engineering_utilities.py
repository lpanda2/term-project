import pandas as pd
import numpy as np
import datetime as dt
import os
import functools
from collections import defaultdict

_g = globals()

pd.options.display.max_rows = 200000
pd.options.display.max_columns = 500

now = dt.datetime.now().strftime('%Y-%m-%d-%H-%M')
result_dir = f'results/{now}/'
print(now)
# os.makedirs(result_dir, exist_ok=True)

def printer(x):
    print(f'{x}\n-------------------')

prod_data = ['d_items', 'chartevents', 'admissions', 'prescriptions', 
             'diagnoses_icd', 'd_icd_diagnoses', 'patients', 'icustays', 'cptevents']


def read_dev_data(fname):
    datadir = 'demo-data/'
    table_name = fname[:-4]
        
    data = pd.read_csv(f'data/{fname}', dtype=str, encoding='latin1')
    print(table_name, data.shape)
    _g[table_name] = data


def read_all_dev_data():
    for fname in [x for x in os.listdir(datadir) if '.csv' in x]:    
        read_dev_data(fname)
   

def read_prod_data(table_name, chunksize=None):
    if chunksize:
        data = '1'
        return data

    datadir = 'data/'
    fname = table_name.upper() + '.csv.gz'
    data = pd.read_csv(f'data/{fname}', dtype=str, encoding='latin1', 
                       compression='gzip')
    data.columns = [x.lower() for x in data.columns]
    print(table_name, data.shape)
    return data


def read_crosswalk():
    return pd.read_csv('code_descriptions/icd2hccxw2014.csv', dtype='str')


def add_aki_hcc_label(diagnoses, icdxw, admissions):
    """take the diagnoses dataframe and create the beginning of 
    a labeled dataset"""
    merged = pd.merge(diagnoses, icdxw, how='left', 
                      left_on='icd9_code', right_on='icd', 
                      indicator='_hcc_merge', validate='m:1')

    # remove them from df and make final labeled dataframe
    merged = pd.concat([merged, 
                        pd.get_dummies(merged.hcc, prefix='hcc_cd')], axis=1)
    cols = [x for x in merged if 'hcc_cd' in x]
    data = merged.groupby(['hadm_id', 'subject_id'], as_index=False)[cols].max()
    
    drop_cols = [x for x in cols if '_135' not in x]
    data = data.drop(drop_cols, axis=1)
    
    # merge in admissions
    data = data.merge(admissions[['hadm_id', 'subject_id', 
                                  'admittime', 'dischtime']], 
                      how='left', on=['hadm_id', 'subject_id'])
    
    # trasnform datatypes
    for i in [x for x in data if 'time' in x]:
        data[i] = pd.to_datetime(data[i], errors='coerce')

    return data
    

def make_clean_charts_data(chartevents, d_items, label):

    # merge in labels
    charts = chartevents.merge(d_items, how='left', on='itemid', indicator='_d_items')
    f'charts data shape: {charts.shape}'
    
    if (charts._d_items != 'both').any():
        printer('\nmerge statistics')
        print(charts._d_items.value_counts())

    # convert time fields to datetime
    for col in ['charttime', 'storetime']:
        charts[col] = pd.to_datetime(charts[col], errors='coerce')

    # use the earliest time between events that are recorded directly in the chart
    # and events that are manually stored
    charts['eventtime'] = charts[['charttime', 'storetime']].min(axis=1)
    
    # change the valuenum field to numeric in case we need it
    charts['valuenum'] = pd.to_numeric(charts['valuenum'], errors='coerce')
    
    # drop unnecessary columns
    drop_cols = ['conceptid', 'param_type', '_d_items', 'valueuom', 'warning',
                'error', 'resultstatus', 'stopped', 'row_id_x', 'row_id_y',
                'linksto']
    charts = charts.drop(drop_cols, axis=1)
    
    # make this upper case to avoid issues with spelling and capitalization
    charts['category'] = charts['category'].str.lower()\
                                           .str.replace('-', '')\
                                           .str.replace('  ', '')\
                                           .str.replace(' ', '_')\
                                           .str.replace(r"\'s", '')\
                                           .str.replace('\/', '_')\
                                           .str.replace('(', '')\
                                           .str.replace(')', '')

    charts['category'] = charts['category']
    
    # merge in label
    charts = charts.merge(label, how='left', on=['subject_id', 'hadm_id'])
    return charts


def create_contrast_imaging_feature(cptevents):
    """cpt codes for imaging with contrast dyes"""
    radiology_cpt_codes = [
        '74177',
        '74160',
        '71260',
        '74177',
        '73701',
        '73201',
        '70460',
        '70487',
        '70491',
        '70481',
        '72193',
        '72126',
        '72132',
        '72129',
        '75574',
        '75572',
        '70545',
        '70548'
    ] # codes with contrast from 2019 (i know it's not the right year)

    cptevents['ft_contrast_imaging'] = cptevents.cpt_cd.isin(radiology_cpt_codes)*1
    return cptevents.groupby('hadm_id', as_index=False)['ft_contrast_imaging'].max()


def add_nephrotoxin_features(prescriptions, admissions):
    """add features for some drugs"""
    res = prescriptions.merge(admissions, how='left', on=['hadm_id', 'subject_id'])
    meds_list = {}
    meds_list['antibiotics'] = ['bacitracin', 
                                'vancomycin', 
                                'amphotericin', 
                                'cephalexin',
                                'cefadroxil',
                                'tobramycin',
                                'gentamicin',
                                'neomycin',
                                'ciprofloxacin']
    meds_list['blood_pressure'] = ['lisinopril', 
                                   'ramipril', 
                                   'metoprolol', 
                                   'candesartan', 
                                   'valsartan', 
                                   'warfarin']
    meds_list['diuretic'] = ['furosemide', 'torsemide']
    meds_list['nsaid'] = ['ibuprofen', 'naproxen']
                          # , 'ketoprofen']
    meds_list['ulcer'] = ['cimetidine']
    meds_list['other'] = ['propofol']

    res['time_delta'] = pd.to_datetime(res.startdate, errors='coerce') - pd.to_datetime(res.admittime, errors='coerce')
    def within_x_hours(data, x):
        return data.time_delta < pd.Timedelta(x, 'hr')

    flatten = lambda l: [item for sublist in l for item in sublist]

    # all drugs
    drug = pd.Series([False for i in range(len(res))])
    for med in flatten(list(meds_list.values())):
        drug |= res.drug.str.lower().str.contains(med, na=False)

    res['ft_any_nephrotoxin_rx'] = drug*1
    for hr in [24, 48, 72]:
        res[f'ft_any_nephrotoxin_rx_within_{hr}'] = (drug & within_x_hours(res, hr))*1

    # groups of drugs
    for group, drugs in meds_list.items():
        print('\t', group)
        drug = pd.Series([False for i in range(len(res))])
        for med in drugs:
            print('\t\t', med)
            this_drug = res.drug.str.lower().str.contains(med, na=False)
            drug |= this_drug        # add to the large list
            res[f'ft_nephrotoxin_{med}_rx'] = this_drug*1  # make its own feature
            for hr in [24, 48, 72]:
                print('\t\t ', hr)
                res[f'ft_nephrotoxin_{med}_rx_within_{hr}'] = (this_drug & within_x_hours(res, hr))*1

        # any drug in the group
        res[f'ft_nephrotoxin_{group}_rx'] = drug*1
        for hr in [24, 48, 72]:
            res[f'ft_nephrotoxin_{group}_rx_within_{hr}'] = (drug & within_x_hours(res, hr))*1

    features = [x for x in res if 'ft_' in x]
    return res.groupby('hadm_id', as_index=False)[features].max()


def create_prior_admissions(admissions, icustays):
    """prior admissions"""
    # self merge
    res = admissions.merge(admissions, how='left', on=['subject_id'], suffixes=['_first', '_second'])
    
    # change the datatypes
    times = [x for x in res if 'time' in x]
    for i in times:
        res[i] = pd.to_datetime(res[i], errors='coerce')
    
    # remove comparison with self
    res = res.loc[res.hadm_id_first != res.hadm_id_second]
    res = res.rename(columns={'hadm_id_first': 'hadm_id'})
    
    # add icu data
    res = res.merge(icustays, how='left', left_on='hadm_id_second', right_on='hadm_id', suffixes=['', '_icu'])
    res = pd.concat([res, pd.get_dummies(res.last_careunit.str.lower(), dtype=bool)], axis=1)
    
    # make features
    prior_admission_30 = (res.admittime_second - res.dischtime_first).dt.days.lt(30)
    prior_admission_60 = (res.admittime_second - res.dischtime_first).dt.days.lt(60)
    prior_admission_90 = (res.admittime_second - res.dischtime_first).dt.days.lt(90)
    prior_admission_120 = (res.admittime_second - res.dischtime_first).dt.days.lt(120)
    
    res['ft_prior_admission_30'] = prior_admission_30*1
    res['ft_prior_admission_60'] = prior_admission_60*1
    res['ft_prior_admission_90'] = prior_admission_90*1
    res['ft_prior_admission_120'] = prior_admission_120*1
    
    res['ft_avg_icu_los_within_30'] = np.where(prior_admission_30, res.los.astype(float), np.nan)
    res['ft_micu_within_30'] = (res.micu & prior_admission_30) * 1
    res['ft_ccu_within_30'] = (res.ccu & prior_admission_30) * 1
    
    features = [x for x in res if 'ft_' in x]
    return res.groupby('hadm_id', as_index=False)[features].max()

def make_labs_data(labevents, d_labitems):
    labs = labevents.merge(d_labitems, how='left', on='itemid')
    labs = labs[['subject_id', 'hadm_id', 'charttime', 'value',
                 'valuenum', 'valueuom', 'label', 'fluid', 'category', 'flag']]
    labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
    labs['label'] = labs.label.str.lower()
    labs['fluid'] = labs.fluid.str.lower()
    labs['category'] = labs.category.str.lower()
    labs['charttime'] = pd.to_datetime(labs.charttime)
    labs = labs.dropna(subset=['hadm_id'])
    return labs

def create_hcc_feature(hccs, label='', rename_as=None):
    """select an hcc feature"""
    cols = [x for x in hccs if 'hcc_' in x]
    if label:
        drop_cols = [x for x in cols if x != 'hcc_cd' + label]
        hccs = hccs.drop(drop_cols, axis=1)
    
    if rename_as:
        assert isinstance(rename_as, str)
        hccs = hccs.rename(columns={'hcc_cd' + label: rename_as})
    return hccs.drop('subject_id', axis=1)
    
def create_hcc_labeled_dataset(diagnoses, icdxw):
    """take the diagnoses dataframe and create the beginning of 
    a labeled dataset"""
    merged = pd.merge(diagnoses, icdxw, how='left', 
                      left_on='icd9_code', right_on='icd', 
                      indicator='_hcc_merge', validate='m:1')
            
    # remove them from df and make final labeled dataframe
    merged = pd.concat([merged, 
                        pd.get_dummies(merged.hcc, prefix='hcc_cd')], axis=1)
    cols = [x for x in merged if 'hcc_cd' in x]
    data = merged.groupby(['hadm_id', 'subject_id'], as_index=False)[cols].max()
    return data

def create_sodium_feature(labs):
    res = labs.loc[labs.label.str.lower().str.contains('sodium', na=False) &
                   (labs.fluid == 'blood'), 
               ['hadm_id', 'charttime', 'label', 'value', 'valuenum']]
    res['ft_low_sodium'] = (res.valuenum < 136) * 1
    return res.groupby('hadm_id', as_index=False)['ft_low_sodium'].max()

def create_potassium_feature(labs):
    res = labs.loc[labs.label.str.lower().str.contains('potassium', na=False) &
                   (labs.fluid == 'blood'), 
               ['hadm_id', 'charttime', 'label', 'value', 'valuenum']]
    res['ft_high_potassium'] = (res.valuenum > 5) * 1
    return res.groupby('hadm_id', as_index=False)['ft_high_potassium'].max()


def create_blood_ph_features(charts):
    """create features for the low blood ph"""
    res = charts.loc[charts.label.str.contains('pH', na=False) | charts.label.str.contains('PH', na=False), 
                     ['hadm_id', 'eventtime', 'admittime', 'label', 
                      'value', 'valuenum', 'unitname']]
    res['label'] = 'blood_ph'
    
    # what is low blood ph
    lowbloodph = res.valuenum.lt(7.30)
    
    res['time_delta'] = res.eventtime - res.admittime    
    def within_x_hours(data, x):
        return data.time_delta < pd.Timedelta(x, 'hr')
    
    res['ft_low_blood_ph'] = lowbloodph*1
    res['ft_low_blood_ph_within_6_hrs'] = (lowbloodph & within_x_hours(res, 6))*1
    res['ft_low_blood_ph_within_12_hrs'] = (lowbloodph & within_x_hours(res, 12))*1
    res['ft_low_blood_ph_within_24_hrs'] = (lowbloodph & within_x_hours(res, 24))*1
    res['ft_low_blood_ph_within_36_hrs'] = (lowbloodph & within_x_hours(res, 36))*1
    res['ft_low_blood_ph_within_48_hrs'] = (lowbloodph & within_x_hours(res, 48))*1
    
    features = [x for x in res if 'ft_' in x]
    return res.groupby('hadm_id', as_index=False)[features].max()


def create_demographics_features(admissions, patients):
    """check admission information and patient demographics"""
    pt = admissions.merge(patients, how='left', on='subject_id')

    # convert data types
    pt['dob'] = pd.to_datetime(pt.dob, errors='coerce')
    pt['admittime'] = pd.to_datetime(pt.admittime, errors='coerce')

    # remap gender as a binary variable
    pt['ft_gender'] = pt.gender.map({'F': 0, 'M': 1})

    # create an age feature
    pt['ft_age'] = pt.admittime.sub(pt.dob, axis=0).dt.days / 365.25
    pt['ft_age'] = np.where(pt.ft_age < -1, np.nan, pt.ft_age) 
    # null out fake dobs, they don't give us information

    # admit type feature
    admit_type = pd.concat([pt.hadm_id, 
                            pd.get_dummies(pt['admission_type'].str.lower(),
                                        prefix='ft_admit_type')], axis=1)

    # ethnicity feature
    pt['ethnicity'] = pt['ethnicity'].str.lower()\
                                         .str.replace('/', '_')\
                                             .str.replace(' ', '_')\
                                                 .str.replace('-', '')\
                                                 .str.replace('__', '_')
    ethnicity = pd.concat([pt.hadm_id, 
                           pd.get_dummies(pt['ethnicity'], 
                                            prefix='ft_race')], axis=1)
    
    combine_these = ['ft_race_patient_declined_to_answer',
                     'ft_race_unable_to_be_obtained',
                     'ft_race_unknown_not_specified']
    combine_if_there = [x for x in combine_these if x in ethnicity]
    ethnicity['ft_race_missing_info'] = ethnicity[combine_if_there].max(axis=1)
    ethnicity = ethnicity.drop(combine_if_there, axis=1)
    
    data = pt[['hadm_id', 'ft_age', 'ft_gender']].merge(admit_type, 
                                                        how='left',
                                                        on='hadm_id')
    data = data.merge(ethnicity, how='left', on='hadm_id')    
    
    data = pt[['hadm_id', 'ft_age', 'ft_gender']].merge(admit_type, 
                                                        how='left',
                                                        on='hadm_id')
    data = data.merge(ethnicity, how='left', on='hadm_id')
    
    agg_dict = {'ft_age': 'mean', 'ft_gender': 'first'}
    agg_dict.update({k:'max' for k in admit_type if k != 'hadm_id'})
    agg_dict.update({k:'max' for k in ethnicity if k != 'hadm_id'})
    
    return data.groupby('hadm_id', as_index=False).agg(agg_dict)


def create_blood_pressure_features(charts):
    """check the hypertensive & hypotensive status"""
    res = charts.loc[charts.label.str.lower().str.contains('diastolic', na=False), 
                     ['hadm_id', 'eventtime', 'admittime', 'label', 
                      'value', 'valuenum', 'unitname']]
    res = res.loc[~res.label.str.lower().str.contains('unloading', na=False)] # remove the apache
    res = res.loc[~res.label.str.lower().str.contains('pulmonary', na=False)] # remove the apache
    res = res.loc[~res.label.str.lower().str.contains('pap', na=False)] # remove the apache
    res.label = 'diastolic_blood_pressure'
    
    res2 = charts.loc[charts.label.str.lower().str.contains('systolic', na=False), 
                     ['hadm_id', 'eventtime', 'admittime', 'label', 
                      'value', 'valuenum', 'unitname']]
    res2 = res2.loc[~res2.label.str.lower().str.contains('unloading', na=False)] # remove the apache
    res2 = res2.loc[~res2.label.str.lower().str.contains('pulmonary', na=False)] # remove the apache
    res2 = res2.loc[~res2.label.str.lower().str.contains('pap', na=False)] # remove the apache
    res2.label = 'systolic_blood_pressure'
    
    # create combined events
    data = res2.merge(res, how='outer', on=['hadm_id', 'eventtime'], indicator=True, 
                 suffixes=['_systolic', '_diastolic'])
    data = data.loc[data._merge == 'both'].drop('_merge', axis=1)

    
    # delete stuff to clear memory
    del res
    del res2
    
    # make features
    elevated = data.valuenum_systolic.between(120, 129) & data.valuenum_diastolic.lt(80)
    abnormally_low = data.valuenum_systolic.lt(119) & data.valuenum_diastolic.lt(79)
    hbp_stg_1 = data.valuenum_systolic.between(130, 139) | data.valuenum_diastolic.between(80, 89)
    hbp_stg_2 = data.valuenum_systolic.between(140, 179) | data.valuenum_diastolic.between(90, 119)
    crisis = data.valuenum_systolic.gt(180) | data.valuenum_diastolic.gt(120)
    
    # hours since admission
    data['time_delta'] = data.eventtime - data.admittime_systolic
    
    def within_x_hours(data, x):
        return data.time_delta < pd.Timedelta(x, 'hr')
    
    data['ft_elevated_bp'] = elevated*1
    data['ft_abnormally_low_bp'] = abnormally_low*1
    data['ft_hbp_stg_1'] = hbp_stg_1*1
    data['ft_hbp_stg_2'] = hbp_stg_2*1
    data['ft_hbp_crisis'] = crisis*1
    
    data['ft_hbp_stg_2_within_6_hours'] = (hbp_stg_2 & within_x_hours(data, 6)) * 1
    data['ft_hbp_stg_2_within_12_hours'] = (hbp_stg_2 & within_x_hours(data, 12)) * 1
    data['ft_hbp_stg_2_within_24_hours'] = (hbp_stg_2 & within_x_hours(data, 24)) * 1
    data['ft_hbp_stg_2_within_36_hours'] = (hbp_stg_2 & within_x_hours(data, 36)) * 1
    data['ft_hbp_stg_2_within_48_hours'] = (hbp_stg_2 & within_x_hours(data, 48)) * 1
    
    features = [x for x in data if 'ft_' in x]
    data = data.groupby('hadm_id', as_index=False)[features].max()
    return data


def create_hematocrit_features(charts, pt):
    """ check hematocrit and hemoglobin levels for anemia """
    res = charts.loc[charts.label.str.lower().str.contains('hematocrit', na=False), 
                     ['hadm_id', 'eventtime', 'admittime', 'label', 'value', 'valuenum', 'unitname']]
    res = res.loc[~res.label.str.lower().str.contains('apache', na=False)] # remove the apache

    # add the gender
    res = res.merge(pt[['hadm_id', 'ft_gender']], how='left', on='hadm_id')

    # clean the label name
    res['label'] = 'hematocrit'

    # add features
    male = res.ft_gender == 1
    male_range = (42, 50)
    female_range = (37, 47)
    above_normal = (male & (res.valuenum > male_range[1])) | (~male & (res.valuenum > female_range[1]))
    below_normal = (male & (res.valuenum < male_range[0])) | (~male & (res.valuenum < female_range[0]))
    way_below_normal = res.valuenum < 20

    res['ft_avg_hematocrit'] = res.valuenum
    res['ft_above_normal_hematocrit'] = above_normal*1
    res['ft_below_normal_hematocrit'] = below_normal*1
    res['ft_way_below_normal_hematocrit'] = way_below_normal*1
    res = res.drop('ft_gender', axis=1)
    
    agg = {'ft_avg_hematocrit': 'mean',
          'ft_above_normal_hematocrit': 'max',
          'ft_below_normal_hematocrit': 'max',
          'ft_way_below_normal_hematocrit': 'max'}

    return res.groupby('hadm_id', as_index=False).agg(agg)


def create_creatinine_features(charts, test=False):
    """creates features for creatinine """
    # make a dataframe of just creatinine data
    res = charts.loc[charts.label.str.lower().str.contains('creatin', na=False), 
               ['hadm_id', 'eventtime', 'admittime', 'label', 
                'value', 'valuenum', 'unitname']]
    
    # drop crazy values
    res = res.loc[res.valuenum <= 11]
    
    # sort by hospital admission and event time
    res = res.sort_values(['hadm_id', 'eventtime']).reset_index(drop=True)

    # get the value of the old test and compare to current test
    res['value_of_previous_test'] = np.where(
        res.hadm_id == res.hadm_id.shift(1),
        res.valuenum.shift(1), res.valuenum)
    res['delta'] = res.valuenum - res.value_of_previous_test

    # get time previous test was administered and compare to current time
    res['time_of_previous_test'] = np.where(
        res.hadm_id == res.hadm_id.shift(1), 
        res.eventtime.shift(1), res.eventtime)
    res['delta_time'] = res.eventtime - res.time_of_previous_test

    # check if time is within a certain range
    res['within_48'] = res.delta_time <= pd.Timedelta(48, 'h')
    res['baseline_creat'] = res.groupby('hadm_id')['valuenum'].transform('first')

    # make features
    res['ft_creatinine_increase_within_48'] = ((res.delta >= 0.3) & res.within_48)*1
    res['ft_creatinine_increase_from_baseline'] = (res.valuenum >= 1.5*res.baseline_creat)*1
    res['ft_baseline_creat_gt_1'] = (res.baseline_creat > 1) * 1
    res['ft_avg_creatinine'] = res.valuenum
    res['ft_baseline_creatinine'] = res.baseline_creat

    if test:
        return res.loc[res.groupby('hadm_id')['ft_creatinine_increase_within_48'].transform('max') == 1, 
                        ['hadm_id', 'valuenum', 'delta', 'delta_time',
                         'baseline_creat',
                         'ft_creatinine_increase_within_48', 
                         'ft_creatinine_increase_from_baseline',
                         'ft_baseline_creat_gt_1']]
    
    features = [x for x in res if 'ft_' in x]
    return res.groupby('hadm_id', as_index=False).agg({
                    'ft_creatinine_increase_within_48': 'max',
                    'ft_creatinine_increase_from_baseline': 'max',
                    'ft_baseline_creat_gt_1': 'max',
                    'ft_baseline_creatinine': 'mean',
                    'ft_avg_creatinine': 'mean'})


def create_urine_features(charts):
    """ features of urine color and appearance"""
    res = charts.loc[charts.label.str.lower().str.contains('urine', na=False), 
               ['hadm_id', 'eventtime', 'admittime', 'label',
                'value', 'valuenum', 'unitname']]

    # clean the label column
    res['label'] = res.label.str.replace('[', '').str.replace(']', '')

    # urine color
    label = 'Urine Color'
    color = res.loc[res.label == label]
    color = pd.concat([color.hadm_id, pd.get_dummies(color.value.str.lower(), 
                                                 prefix=('ft_' + label.lower().replace(' ', '_')))],
                  axis=1)

    # urine appearance
    label = 'Urine Appearance'
    appearance = res.loc[res.label == label]
    appearance = pd.concat([appearance.hadm_id,
                            pd.get_dummies(appearance.value.str.lower(), 
                                prefix=('ft' + label.lower().replace(' ', '_')))],
                  axis=1)

    del res

    data = color.merge(appearance, how='outer', on='hadm_id')

    del appearance
    del color

    data = data.groupby('hadm_id', as_index=False)[[x for x in data if 'ft_' in x]].max()
    return data


def merge_features(feature_list):
    """ merge dataframes of features that have hadm_id """
    return functools.reduce(lambda x,y: pd.merge(x,y, how='outer', on='hadm_id'), feature_list)

def read_charts_data(bin_id):
    return pd.read_csv(f'split-data/chartevents/bin_{bin_id}.csv',
                       dtype=str,
                       nrows=10**6)

def create_mechanical_ventilation_feature(cptevents):
    mechanical = cptevents.cpt_cd == '94003'
    mechanical &= cptevents.description.str.lower().str.contains('invasive', na=False)
    cptevents['ft_mechanical_ventilation'] = mechanical*1
    return cptevents.groupby('hadm_id', as_index=False)['ft_mechanical_ventilation'].max()

def create_anemia_feature(labs):
    res = labs.loc[(labs.label == 'iron') &
                   (labs.fluid == 'blood'), 
               ['hadm_id', 'charttime', 'label', 'value', 'valuenum']]
    res['ft_anemic'] = res.valuenum < 50
    return res.groupby('hadm_id', as_index=False)['ft_anemic'].max()  


def charts_data_wrapper(bin_id, d_items, df, demographic_features, i):

    print_me = i % 20 == 0
    print('charts chunk: ', i)

    chartevents = read_charts_data(bin_id)
    if print_me:
        print('chartevents', chartevents.shape)
    charts = make_clean_charts_data(chartevents, d_items, df)
    charts = charts.drop_duplicates(['hadm_id', 'itemid', 'eventtime'])

    creatinine_features = create_creatinine_features(charts)
    if print_me:
        printer('creatinine features')
        print(creatinine_features.shape)

    #urine_features = create_urine_features(charts)
    #if print_me:
    #    printer('urine features')
    #    print(urine_features.shape)

    hematocrit_features = create_hematocrit_features(charts, demographic_features)
    if print_me:
        printer('hematocrit features')
        print(hematocrit_features.shape)

    hypertensive_features = create_blood_pressure_features(charts)
    if print_me:
        printer('hypertensive features')
        print(hypertensive_features.shape)
    
    blood_ph_features = create_blood_ph_features(charts)
    if print_me:
        printer('blood ph features')
        print(blood_ph_features.shape)

    return merge_features([
        creatinine_features,
        #urine_features,
        hematocrit_features,
        hypertensive_features,
        blood_ph_features])

def make_bar(data):
    ax = data.hcc_cd_135.hist(grid=False, bins=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([0,1])
    ax.bar(data.hcc_cd_135.value_counts().index, 
           data.hcc_cd_135.value_counts().values, 0.5, align='center')

