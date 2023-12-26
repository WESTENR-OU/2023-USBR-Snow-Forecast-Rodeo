import glob
import os
import sys

import pandas as pd
import warnings
import Utils
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np
from datetime import datetime
import pdb

warnings.filterwarnings('ignore')
random_state = 42
# **************************************************************************** #
# Read in naturalized streamflow (target variable)
# **************************************************************************** #
# Train set
file_path = 'train_1128update.csv'
df = pd.read_csv(file_path)
df = df.rename(columns={'year': 'WY'})
df['site_id_short'] = df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)
# Group the DataFrame by 'site_id'
# Initialize max_records and min_records before the loop
max_records = 0
min_records = float('inf')  # Set min_records to positive infinity initially

# Create a dictionary to store DataFrames by site_id_short
df_by_sites = {}
# Group by 'site_id_short'
grouped_dataframes = df.groupby('site_id_short')
# Loop through each group
for site_id_short, group_df in grouped_dataframes:
    # Store the DataFrame in df_by_sites
    df_by_sites[site_id_short] = group_df
    # Drop NaN values in-place
    df_by_sites[site_id_short].dropna(inplace=True)
    # Update max_records and min_records based on the length of the DataFrame
    current_records = len(df_by_sites[site_id_short])
    if current_records > max_records:
        max_records = current_records
    if current_records < min_records:
        min_records = current_records

# **************************************************************************** #
# Sites and forecast date
# **************************************************************************** #

site_id_short = ['HHI', 'SRH', 'PRI', 'SRA', 'MRT', 'ARD', 'YRM', 'LRI', 'BRB',
                 'GRD', 'TPI', 'DRI', 'RRI', 'FRI', 'WRO', 'SJR', 'MRB', 'ARL',
                 'CRF', 'SRS', 'DLI', 'VRV', 'SRR', 'BRI', 'PRP', 'ORD']  #

site_id_in_monthlyNSF = pd.read_csv('train_monthly_naturalized_flow_1128update.csv')['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x) for x in site_id_in_monthlyNSF]

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

test_year = ['2005', '2007', '2009', '2011', '2013', '2015', '2017', '2019', '2021', '2023']

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

# **************************************************************************** #
# Smoke submission initialization
# **************************************************************************** #
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
for date in submission_issue_date:
    print(f'forecast date: {date}')
    concatenated_PPT = []
    concatenated_SWE = []
    concatenated_Tmax = []
    concatenated_Tmin = []
    concatenated_NSF = []
    concatenated_monthly_NSF = []
    if date[:1] == '07':
        site_id_short.remove('DLI')  # DLI only to June
    for site in site_id_short_in_monthlyNSF:  # TODO 23 SITES FOR MONTHLY NSF
        # Predictors
        # Read in the predictors
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

        # process the predictors
        PPT_mean_df = Utils.compute_monthly_mean(PPT_df, date,
                                                 PPT_folder_path + '/%s_%s_monthly_mean.csv' % (site, date))
        SWE_mean_df = Utils.compute_monthly_mean(SWE_df, date,
                                                 SWE_folder_path + '/%s_%s_monthly_mean.csv' % (site, date),
                                                 swe=True)
        Tmax_mean_df = Utils.compute_monthly_mean(Tmax_df, date,
                                                  Tmax_folder_path + '/%s_%s_monthly_mean.csv' % (site, date))
        Tmin_mean_df = Utils.compute_monthly_mean(Tmin_df, date,
                                                  Tmin_folder_path + '/%s_%s_monthly_mean.csv' % (site, date))

        # Slice both predictors and predictant to desired period (defined earlier)
        NSF_df = Utils.slice_df(df_by_sites[site], start_year, end_year)
        PPT_df_tr = Utils.slice_df(PPT_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'PPT_past3',
                                                                                      'Mean_past2': 'PPT_past2',
                                                                                      'Mean_past1': 'PPT_past1'})
        SWE_df_tr = Utils.slice_df(SWE_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'SWE_past3',
                                                                                      'Mean_past2': 'SWE_past2',
                                                                                      'Mean_past1': 'SWE_past1'})
        Tmax_df_tr = Utils.slice_df(Tmax_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmax_past3',
                                                                                        'Mean_past2': 'Tmax_past2',
                                                                                        'Mean_past1': 'Tmax_past1'})
        Tmin_df_tr = Utils.slice_df(Tmin_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmin_past3',
                                                                                        'Mean_past2': 'Tmin_past2',
                                                                                        'Mean_past1': 'Tmin_past1'})
        NSF_train_df = Utils.slice_df(NSF_train_df, start_year, end_year)
        NSF_test_df = Utils.slice_df(NSF_test_df, start_year, end_year)

        ###
        if int(date[0:2]) != 7:
            tab = np.where(NSF_train_df['month'] == int(date[0:2]))
            tab1 = [element for original_element in tab for element in
                    [original_element - 1, original_element - 2, original_element - 3]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            NSF_train_values_tab = np.array(NSF_train_df['Value'])[tab2]
            NSF_train_values_tab = NSF_train_values_tab.reshape(32, 3)
            ###
            tab = np.where(NSF_test_df['month'] == int(date[0:2]))
            tab1 = [element for original_element in tab for element in
                    [original_element - 1, original_element - 2, original_element - 3]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            NSF_test_values_tab = np.array(NSF_test_df['Value'])[tab2]
            NSF_test_values_tab = NSF_test_values_tab.reshape(10, 3)

        elif int(date[0:2]) == 7:
            tab = np.where(NSF_train_df['month'] == int(date[0:2]) - 1)
            tab1 = [element for original_element in tab for element in
                    [original_element, original_element - 1, original_element - 2]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            NSF_train_values_tab = np.array(NSF_train_df['Value'])[tab2]
            NSF_train_values_tab = NSF_train_values_tab.reshape(32, 3)
            ###
            tab = np.where(NSF_test_df['month'] == int(date[0:2]) - 1)
            tab1 = [element for original_element in tab for element in
                    [original_element, original_element - 1, original_element - 2]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            # if date == '07-01':
            #    pdb.set_trace()
            NSF_test_values_tab = np.array(NSF_test_df['Value'])[tab2]
            NSF_test_values_tab = NSF_test_values_tab.reshape(10, 3)

        if NSF_train_df['Value'].isna().any():
            break

        # NSF_df_test = pd.DataFrame(columns=['site_id', 'WY', 'NSF_past3', 'NSF_past2', 'NSF_past1'])
        NSF_df_tr = pd.DataFrame(NSF_train_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
        NSF_df_tr.insert(0, 'site_id', Utils.get_site_full_id(site))
        NSF_df_tr.insert(1, 'WY', NSF_train_df['WY'].unique())

        NSF_df_test = pd.DataFrame(NSF_test_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
        NSF_df_test.insert(0, 'site_id', Utils.get_site_full_id(site))
        NSF_df_test.insert(1, 'WY', NSF_test_df['WY'].unique())

        monthly_NSF_df_tr = pd.concat([NSF_df_tr, NSF_df_test], ignore_index=True).sort_values(by='WY')

        concatenated_PPT.append(PPT_df_tr)
        concatenated_SWE.append(SWE_df_tr)
        concatenated_Tmax.append(Tmax_df_tr)
        concatenated_Tmin.append(Tmin_df_tr)
        concatenated_NSF.append(NSF_df)
        concatenated_monthly_NSF.append(monthly_NSF_df_tr)

    # Concatenate all DataFrames for different sites into one
    # Regional training
    PPT_df_tr = pd.concat(concatenated_PPT, ignore_index=True)
    SWE_df_tr = pd.concat(concatenated_SWE, ignore_index=True)
    Tmax_df_tr = pd.concat(concatenated_Tmax, ignore_index=True)
    Tmin_df_tr = pd.concat(concatenated_Tmin, ignore_index=True)
    monthly_NSF_df_tr = pd.concat(concatenated_monthly_NSF, ignore_index=True)

    target_NSF_df_tr = pd.concat(concatenated_NSF, ignore_index=True)

    # **************************************************************************** #
    # Train test data define
    # **************************************************************************** #
    input_df = pd.concat([PPT_df_tr,
                          SWE_df_tr.drop(columns=['site_id', 'WY']),
                          Tmax_df_tr.drop(columns=['site_id', 'WY']),
                          Tmin_df_tr.drop(columns=['site_id', 'WY']),
                          monthly_NSF_df_tr.drop(columns=['site_id', 'WY'])],
                         axis=1)

    merged_df = pd.merge(input_df, target_NSF_df_tr, on=['WY', 'site_id'])

    X = merged_df[['PPT_past3', 'PPT_past2', 'PPT_past1',
                   'SWE_past3', 'SWE_past2', 'SWE_past1',
                   'Tmax_past3', 'Tmax_past2', 'Tmax_past1',
                   'Tmin_past3', 'Tmin_past2', 'Tmin_past1',
                   'NSF_past3', 'NSF_past2', 'NSF_past1']]
    y = merged_df['volume']

    train_df = merged_df[merged_df['WY'].isin(year_list[:28])]  # TODO DELETE IF NOT VAL OFFLINE
    X = train_df[['PPT_past3', 'PPT_past2', 'PPT_past1',
                  'SWE_past3', 'SWE_past2', 'SWE_past1',
                  'Tmax_past3', 'Tmax_past2', 'Tmax_past1',
                  'Tmin_past3', 'Tmin_past2', 'Tmin_past1',
                  'NSF_past3', 'NSF_past2', 'NSF_past1']]  # TODO DELETE IF NOT VAL OFFLINE
    y = train_df['volume']  # TODO DELETE IF NOT VAL OFFLINE
    # Split the data into training and testing sets
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0, random_state=random_state)
    X_train = X
    y_train = y

    test_df = input_df[input_df['WY'].isin(submission_test_year)]
    X_test = test_df[['PPT_past3', 'PPT_past2', 'PPT_past1',
                      'SWE_past3', 'SWE_past2', 'SWE_past1',
                      'Tmax_past3', 'Tmax_past2', 'Tmax_past1',
                      'Tmin_past3', 'Tmin_past2', 'Tmin_past1',
                      'NSF_past3', 'NSF_past2', 'NSF_past1']]

    test_df = merged_df[merged_df['WY'].isin(year_list[28:])]  # TODO DELETE IF NOT VAL OFFLINE
    X_test = test_df[['PPT_past3', 'PPT_past2', 'PPT_past1',
                      'SWE_past3', 'SWE_past2', 'SWE_past1',
                      'Tmax_past3', 'Tmax_past2', 'Tmax_past1',
                      'Tmin_past3', 'Tmin_past2', 'Tmin_past1',
                      'NSF_past3', 'NSF_past2', 'NSF_past1']]  # TODO DELETE IF NOT VAL OFFLINE
    y_test = test_df['volume']  # TODO DELETE IF NOT VAL OFFLINE
    site_id_test = test_df['site_id'].unique()  # TODO DELETE IF NOT VAL OFFLINE
    WY_test = test_df['WY'].unique()  # TODO DELETE IF NOT VAL OFFLINE

    # **************************************************************************** #
    # Gradient boost (for the current forecast date)
    # **************************************************************************** #
    # Initialize the Random Forest Regressor
    hyperparameters = {'max_depth': 15,
                       'max_features': None,
                       'min_samples_leaf': 4,
                       'min_samples_split': 10,
                       'n_estimators': 70,
                       'learning_rate': 0.05}
    # Perform quantile regression for each quantile
    GB_regressor_models = {}
    df_temps = {}
    for quantile in quantiles:
        GB_regressor = GradientBoostingRegressor(**hyperparameters, loss='quantile', alpha=quantile,
                                                 random_state=random_state)
        GB_regressor.fit(X_train, y_train)
        df_temp = pd.DataFrame(columns=['site_id', 'WY', 'value'])
        df_temp['site_id'] = test_df['site_id']
        df_temp['WY'] = test_df['WY']
        df_temp['value'] = GB_regressor.predict(X_test)
        GB_regressor_models[quantile] = GB_regressor
        df_temps[quantile] = df_temp
    predictions_GBR = {q: model.predict(X_test) for q, model in GB_regressor_models.items()}
    predictions_GBR = {q: df for q, df in df_temps.items()}
    pred_df_GBR_dict[date] = predictions_GBR

    mean_QL = Utils.mean_quantile_loss(y_test, predictions_GBR, quantiles)
    IC = Utils.calculate_interval_coverage(y_test, predictions_GBR[quantiles[0]]["value"],
                                           predictions_GBR[quantiles[-1]]["value"])

    print(f' GBR Mean quantile loss: {mean_QL}, IC: {IC}')

    mean_QL_df.loc[mean_QL_df['issue_date'] == date, 'GBR'] = mean_QL
    IC_df.loc[IC_df['issue_date'] == date, 'GBR'] = IC

    # **************************************************************************** #
    # XGBoost (for the current forecast date)
    # **************************************************************************** #
    evals_result = {}  # Dict[str, Dict]
    Xy_train = xgb.QuantileDMatrix(X_train, y_train)
    # use Xy as a reference
    # Xy_val = xgb.QuantileDMatrix(X_test, y_test, ref=Xy_train)
    ## XGB
    booster = xgb.train(
        {
            # Use the quantile objective function.
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": np.array(quantiles),
            "learning_rate": 0.04,
            "max_depth": 8,
        },
        Xy_train,
        num_boost_round=32,
        early_stopping_rounds=2,
        # The evaluation result is a weighted average across multiple quantiles.
        evals=[(Xy_train, "Train")],
        evals_result=evals_result)

    scores = booster.inplace_predict(X_test)
    y_lower = scores[:, 0]  # alpha=0.1
    y_med = scores[:, 1]  # alpha=0.5, median
    y_upper = scores[:, 2]  # alpha=0.9
    y_lower_df = pd.DataFrame(columns=['site_id', 'WY', 'value'])
    y_lower_df['site_id'] = test_df['site_id']
    y_lower_df['WY'] = test_df['WY']
    y_lower_df['value'] = y_lower
    y_med_df = pd.DataFrame(columns=['site_id', 'WY', 'value'])
    y_med_df['site_id'] = test_df['site_id']
    y_med_df['WY'] = test_df['WY']
    y_med_df['value'] = y_med
    y_upper_df = pd.DataFrame(columns=['site_id', 'WY', 'value'])
    y_upper_df['site_id'] = test_df['site_id']
    y_upper_df['WY'] = test_df['WY']
    y_upper_df['value'] = y_upper
    predictions_XGB = {0.1: y_lower, 0.5: y_med, 0.9: y_upper}
    predictions_XGB = {0.1: y_lower_df, 0.5: y_med_df, 0.9: y_upper_df}
    pred_df_XGB_dict[date] = predictions_XGB

    mean_QL = Utils.mean_quantile_loss(y_test, predictions_XGB, quantiles)
    IC = Utils.calculate_interval_coverage(y_test, predictions_XGB[quantiles[0]]['value'],
                                           predictions_XGB[quantiles[-1]]['value'])
    print(f' XGB Mean quantile loss: {mean_QL}, IC: {IC}')
    mean_QL_df.loc[mean_QL_df['issue_date'] == date, 'XGB'] = mean_QL
    IC_df.loc[IC_df['issue_date'] == date, 'XGB'] = IC

    # ENSEMBLE
    df_temps = {}
    mean_pred = {key: (predictions_GBR[key]['value'] + predictions_XGB[key]['value']) / 2 for key in predictions_GBR if
                 key in predictions_XGB}
    for quantile in quantiles:
        df_temp = pd.DataFrame(columns=['site_id', 'WY', 'value'])
        df_temp['site_id'] = test_df['site_id']
        df_temp['WY'] = test_df['WY']
        df_temp['value'] = mean_pred[quantile]
        df_temps[quantile] = df_temp
    predictions_mean = {q: df for q, df in df_temps.items()}
    pred_df_mean_dict[date] = predictions_mean

    mean_QL = Utils.mean_quantile_loss(y_test, predictions_mean, quantiles)
    IC = Utils.calculate_interval_coverage(y_test, predictions_mean[quantiles[0]]['value'],
                                           predictions_mean[quantiles[-1]]['value'])
    print(f' Ensemble Mean quantile loss: {mean_QL}, IC: {IC}')
    mean_QL_df.loc[mean_QL_df['issue_date'] == date, 'Ensemble_Mean'] = mean_QL
    IC_df.loc[IC_df['issue_date'] == date, 'Ensemble_Mean'] = IC

QL = mean_QL_df.drop('issue_date', axis=1).mean()
IC = IC_df.drop('issue_date', axis=1).mean()
# Display mean values for each column
for column, mean_value in QL.items():
    print(f"Mean QL of column '{column}': {mean_value}")
for column, mean_value in IC.items():
    print(f"IC of column '{column}': {mean_value}")

'''
# Assign values to the smoke submission file
submission_GBR = submission_file.copy()
submission_XGB = submission_file.copy()
submission_mean = submission_file.copy()

for site_id in submission_site_id:
    for date in submission_issue_date:
        for WY in submission_test_year:
            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.1)
            Utils.update_smoke_submission(submission_XGB, pred_df_XGB_dict, date, site_id, WY, 0.1)
            Utils.update_smoke_submission(submission_mean, pred_df_mean_dict, date, site_id, WY, 0.1)

            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.5)
            Utils.update_smoke_submission(submission_XGB, pred_df_XGB_dict, date, site_id, WY, 0.5)
            Utils.update_smoke_submission(submission_mean, pred_df_mean_dict, date, site_id, WY, 0.5)

            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.9)
            Utils.update_smoke_submission(submission_XGB, pred_df_XGB_dict, date, site_id, WY, 0.9)
            Utils.update_smoke_submission(submission_mean, pred_df_mean_dict, date, site_id, WY, 0.9)

submission_GBR.to_csv('submission_GBR.csv', index=False)
submission_XGB.to_csv('submission_XGB.csv', index=False)
submission_mean.to_csv('submission_mean.csv', index=False)
'''
