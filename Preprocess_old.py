import glob
import os
import pdb

import pandas as pd
import warnings
import Utils
import numpy as np
from datetime import datetime
import json

warnings.filterwarnings('ignore')
random_state = 42

metadata = pd.read_csv('metadata_TdPVeJC.csv')
# Train set
file_path = 'train_1128update.csv'
target_df = pd.read_csv(file_path)
target_df = target_df.rename(columns={'year': 'WY'})
target_df['site_id_short'] = target_df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)

# Group the DataFrame by 'site_id'
# Create a dictionary to store DataFrames by site_id_short
df_by_sites = {}
# Group by 'site_id_short'
grouped_dataframes = target_df.groupby('site_id_short')
# Loop through each group
for site_id_short, group_df in grouped_dataframes:
    # Store the DataFrame in df_by_sites
    df_by_sites[site_id_short] = group_df
    # Drop NaN values in-place
    df_by_sites[site_id_short].dropna(inplace=True)


site_id_short = target_df['site_id_short'].unique()
site_id_in_monthlyNSF = pd.read_csv('train_monthly_naturalized_flow_1128update.csv')['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x) for x in site_id_in_monthlyNSF]
site_id_not_in_monthlyNSF = list(set(target_df['site_id'].unique()) - set(site_id_in_monthlyNSF))

forecast_date = []
days = ['01', '08', '15', '22']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

# **************************************************************************** #
# Define training and testing period
# **************************************************************************** #
start_year = 1982
end_year = 2023
year_list = list(range(start_year, end_year + 1))
quantiles = [0.1, 0.5, 0.9]

# **************************************************************************** #
# Normal submission initialization
# **************************************************************************** #
submission_file = pd.read_csv('submission_format_1128update.csv')
submission_issue_date = []
submission_test_year = []
for date in submission_file['issue_date'].unique():
    submission_issue_date.append(date[-5:])
    submission_test_year.append(date[:4])
submission_issue_date = list(set(submission_issue_date))
# Convert date strings to datetime objects
date_objects = [datetime.strptime(date, '%m-%d') for date in submission_issue_date]
# Sort datetime objects
sorted_dates = sorted(date_objects)
# Convert sorted datetime objects back to date strings
submission_issue_date = [date.strftime('%m-%d') for date in sorted_dates]

submission_test_year = list(set(submission_test_year))
submission_test_year = [int(x) for x in submission_test_year]

submission_site_id = submission_file['site_id'].unique()
submission_site_short_id = []
for site_id in submission_site_id:
    submission_site_short_id.append(Utils.get_site_short_id(site_id))

smoke_submission_file = pd.read_csv('smoke_submission_format_1128update.csv')
smoke_submission_issue_date = []
smoke_submission_test_year = []
for date in smoke_submission_file['issue_date'].unique():
    smoke_submission_issue_date.append(date[-5:])
    smoke_submission_test_year.append(date[:4])
smoke_submission_issue_date = list(set(smoke_submission_issue_date))

# Convert date strings to datetime objects
date_objects = [datetime.strptime(date, '%m-%d') for date in smoke_submission_issue_date]
# Sort datetime objects
sorted_dates = sorted(date_objects)
# Convert sorted datetime objects back to date strings
smoke_submission_issue_date = [date.strftime('%m-%d') for date in sorted_dates]
smoke_submission_test_year = list(set(smoke_submission_test_year))
smoke_submission_test_year = [int(x) for x in smoke_submission_test_year]

smoke_submission_site_id = smoke_submission_file.site_id.unique()
smoke_submission_site_short_id = []
for site_id in smoke_submission_site_id:
    smoke_submission_site_short_id.append(Utils.get_site_short_id(site_id))
# **************************************************************************** #
# Read in and process predictors [PPT_acc, SWE_acc, Tmax_mean, Tmin_mean]
# **************************************************************************** #
mean_QL_df = pd.DataFrame(columns=['issue_date', 'GBR', 'XGB', 'Ensemble_Mean'])
IC_df = pd.DataFrame(columns=['issue_date', 'GBR', 'XGB', 'Ensemble_Mean'])
mean_QL_df['issue_date'] = smoke_submission_issue_date
IC_df['issue_date'] = smoke_submission_issue_date

# Write the prediction for the current issue date
pred_df_GBR = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_XGB = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_mean = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_GBR_dict = {}
pred_df_XGB_dict = {}
pred_df_mean_dict = {}

Data_folder = 'PRISM_USBR_forecast_Rodeo/Updated'

concatenated_PPT = []
concatenated_SWE = []
concatenated_Tmax = []
concatenated_Tmin = []
concatenated_Tmax_var = []
concatenated_Tmin_var = []
concatenated_NSF = []
concatenated_monthly_NSF = []
concatenated_drainage_area = []
for date in submission_issue_date:
    print(f'forecast date: {date}')
    if date[:2] == '07':
        site_id_short = site_id_short[site_id_short != 'DLI']
    for site in site_id_short:
        #if site not in site_id_short_in_monthlyNSF:
        #    continue
        # Predictors
        SWE_folder_path = os.path.join(Data_folder, 'SWE_UA')
        PPT_folder_path = os.path.join(Data_folder, 'PPT')
        Tmax_folder_path = os.path.join(Data_folder, 'T_max')
        Tmin_folder_path = os.path.join(Data_folder, 'T_min')

        SWE_df = Utils.read_predictors(SWE_folder_path + '/%s.csv' % site, site, predictor_is_SWE=True)
        PPT_df = Utils.read_predictors(PPT_folder_path + '/%s.csv' % site, site)
        Tmax_df = Utils.read_predictors(Tmax_folder_path + '/%s_Tmax.csv' % site, site)
        Tmin_df = Utils.read_predictors(Tmin_folder_path + '/%s_Tmin.csv' % site, site)
        NSF_train_df = Utils.read_predictors('train_monthly_naturalized_flow_1128update.csv', site,
                                             predictor_is_NSF=True)
        NSF_test_df = Utils.read_predictors('test_monthly_naturalized_flow_1128update.csv', site,
                                            predictor_is_NSF=True)


        PPT_df, PPT_acc_df = Utils.compute_acc(PPT_df, date, PPT_folder_path + '/%s_%s_acc.csv' % (site, date))
        SWE_df, SWE_acc_df = Utils.compute_acc(SWE_df, date, SWE_folder_path + '/%s_%s_acc.csv' % (site, date),
                                               swe=True)
        Tmax_df, Tmax_mean_df = Utils.compute_acc(Tmax_df, date,
                                                  Tmax_folder_path + '/%s_%s_mean.csv' % (site, date),
                                                  mode='mean')
        Tmin_df, Tmin_mean_df = Utils.compute_acc(Tmin_df, date,
                                                  Tmin_folder_path + '/%s_%s_mean.csv' % (site, date),
                                                  mode='mean')
        #Tmax_var_df = Utils.compute_var(Tmax_df, date)
        #Tmin_var_df = Utils.compute_var(Tmin_df, date)

        # Slice both predictors and predictant to desired period (defined earlier)
        NSF_df = Utils.slice_df(df_by_sites[site], start_year, end_year)
        PPT_df_tr = Utils.slice_df(PPT_acc_df, start_year, end_year).rename(columns={'Acc': 'PPT_acc'})
        SWE_df_tr = Utils.slice_df(SWE_acc_df, start_year, end_year).rename(columns={'Acc': 'SWE_acc'})
        Tmax_df_tr = Utils.slice_df(Tmax_mean_df, start_year, end_year).rename(columns={'Mean': 'Tmax_mean'})
        Tmin_df_tr = Utils.slice_df(Tmin_mean_df, start_year, end_year).rename(columns={'Mean': 'Tmin_mean'})
        #Tmax_var_df_tr = Utils.slice_df(Tmax_var_df, start_year, end_year).rename(columns={'Var': 'Tmax_var'})
        #Tmin_var_df_tr = Utils.slice_df(Tmin_var_df, start_year, end_year).rename(columns={'Var': 'Tmin_var'})
        NSF_train_df = Utils.slice_df(NSF_train_df, start_year, end_year)
        NSF_test_df = Utils.slice_df(NSF_test_df, start_year, end_year)

        NSF_df['WY'] = NSF_df['WY'].astype(str) + '-' + date
        PPT_df_tr['WY'] = PPT_df_tr['WY'].astype(str) + '-' + date
        SWE_df_tr['WY'] = SWE_df_tr['WY'].astype(str) + '-' + date
        Tmax_df_tr['WY'] = Tmax_df_tr['WY'].astype(str) + '-' + date
        Tmin_df_tr['WY'] = Tmin_df_tr['WY'].astype(str) + '-' + date

        ##
        tab = np.where(
            NSF_train_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_train_df['month'] == int(date[0:2]) - 1)
        tab1 = [element for original_element in tab for element in
                [original_element, original_element - 1, original_element - 2]]
        tab2 = np.array(tab1).flatten()
        tab2 = np.sort(tab2)
        NSF_train_values_tab = np.array(NSF_train_df['Value'])[tab2]
        if not pd.Series(NSF_train_values_tab).empty:
            non_nan_indices = np.where(~np.isnan(NSF_train_values_tab))[0]
            # Perform linear interpolation for NaN values
            arr_interp = np.interp(np.arange(len(NSF_train_values_tab)), non_nan_indices, NSF_train_values_tab[non_nan_indices])
            NSF_train_values_tab = arr_interp.reshape(32, 3)
        else:
            NSF_train_values_tab = np.full((32, 3), np.nan)
        ###
        tab = np.where(
            NSF_test_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_test_df['month'] == int(date[0:2]) - 1)
        tab1 = [element for original_element in tab for element in
                [original_element, original_element - 1, original_element - 2]]
        tab2 = np.array(tab1).flatten()
        tab2 = np.sort(tab2)
        NSF_test_values_tab = np.array(NSF_test_df['Value'])[tab2]
        if not pd.Series(NSF_test_values_tab).empty:
            # Indices of non-NaN values
            non_nan_indices = np.where(~np.isnan(NSF_test_values_tab))[0]
            # Perform linear interpolation for NaN values
            arr_interp = np.interp(np.arange(len(NSF_test_values_tab)), non_nan_indices, NSF_test_values_tab[non_nan_indices])
            NSF_test_values_tab = arr_interp.reshape(10, 3)
        else:
            NSF_test_values_tab = np.full((10, 3), np.nan)

        NSF_df_tr = pd.DataFrame(NSF_train_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
        NSF_df_tr.insert(0, 'site_id', Utils.get_site_full_id(site))
        #NSF_df_tr.insert(1, 'WY', NSF_train_df['WY'].unique())
        WY_tr = np.array([1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
              1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
              2004, 2006, 2008, 2010, 2012, 2014,
              2016, 2018, 2020, 2022])
        NSF_df_tr.insert(1, 'WY', WY_tr)
        NSF_df_tr['WY'] = pd.to_datetime(NSF_df_tr['WY'].astype(str) + '-' + date)

        NSF_df_test = pd.DataFrame(NSF_test_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
        NSF_df_test.insert(0, 'site_id', Utils.get_site_full_id(site))
        #NSF_df_test.insert(1, 'WY', NSF_test_df['WY'].unique())
        WY_test = np.array([2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023])
        NSF_df_test.insert(1, 'WY', WY_test)
        NSF_df_test['WY'] = pd.to_datetime(NSF_df_test['WY'].astype(str) + '-' + date)
        monthly_NSF_df_tr = pd.concat([NSF_df_tr, NSF_df_test], ignore_index=True).sort_values(by='WY')

        # get the drainage area of this site
        drainage_area = metadata[metadata['site_id'] == Utils.get_site_full_id(site)]['drainage_area']
        drainage_area_df_tr = PPT_df_tr.copy().rename(columns={'PPT_acc': 'drainage_area'})
        drainage_area_df_tr['drainage_area'] = drainage_area.values[0]

        concatenated_PPT.append(PPT_df_tr)
        concatenated_SWE.append(SWE_df_tr)
        concatenated_Tmax.append(Tmax_df_tr)
        concatenated_Tmin.append(Tmin_df_tr)
        #concatenated_Tmax_var.append(Tmax_var_df_tr)
        #concatenated_Tmin_var.append(Tmin_var_df_tr)
        concatenated_NSF.append(NSF_df)
        concatenated_monthly_NSF.append(monthly_NSF_df_tr)
        concatenated_drainage_area.append(drainage_area_df_tr)

    # Concatenate all DataFrames for different sites into one
    # Regional training
PPT_df_tr = pd.concat(concatenated_PPT, ignore_index=True)
SWE_df_tr = pd.concat(concatenated_SWE, ignore_index=True)
Tmax_df_tr = pd.concat(concatenated_Tmax, ignore_index=True)
Tmin_df_tr = pd.concat(concatenated_Tmin, ignore_index=True)
#Tmax_var_df_tr = pd.concat(concatenated_Tmax_var, ignore_index=True)
#Tmin_var_df_tr = pd.concat(concatenated_Tmin_var, ignore_index=True)
monthly_NSF_df_tr = pd.concat(concatenated_monthly_NSF, ignore_index=True)
drainage_area_df_tr = pd.concat(concatenated_drainage_area, ignore_index=True)

target_NSF_df_tr = pd.concat(concatenated_NSF, ignore_index=True)

input_df = pd.concat([
    PPT_df_tr,
    SWE_df_tr.drop(columns=['site_id', 'WY']),
    Tmax_df_tr.drop(columns=['site_id', 'WY']),
    Tmin_df_tr.drop(columns=['site_id', 'WY']),
    #Tmax_var_df_tr.drop(columns=['site_id', 'WY']),
    #Tmin_var_df_tr.drop(columns=['site_id', 'WY']),
    monthly_NSF_df_tr.drop(columns=['site_id', 'WY']),
    drainage_area_df_tr.drop(columns=['site_id', 'WY'])
],
    axis=1)

for date in submission_issue_date:
    # Extract the month and day parts of the 'WY' column
    input_df['month_day'] = input_df['WY'].str[5:]
    # Filter rows based on the 'month_day' column
    result_df = input_df[input_df['month_day'] == date].copy()

    # Calculate mean values for the specified columns
    NSF_past1_mean = result_df['NSF_past1'].mean()
    NSF_past2_mean = result_df['NSF_past2'].mean()
    NSF_past3_mean = result_df['NSF_past3'].mean()

    # Update specific rows in input_df
    for site_id in site_id_not_in_monthlyNSF:
        condition = (input_df['site_id'] == site_id) & (input_df['month_day'] == date)
        input_df.loc[condition, 'NSF_past1'] = NSF_past1_mean
        input_df.loc[condition, 'NSF_past2'] = NSF_past2_mean
        input_df.loc[condition, 'NSF_past3'] = NSF_past3_mean

# Drop the temporary 'month_day' column
input_df = input_df.drop(columns=['month_day'])

# Save X and y
input_df.to_csv('preprocessed_dir/features_old.csv', index=False)
target_NSF_df_tr.to_csv('preprocessed_dir/target.csv', index=False)
