import glob
import os
import pdb

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

from joblib import dump

import json

warnings.filterwarnings('ignore')
random_state = 42
# **************************************************************************** #
# Use this submission.py in couple with Preprocecss.py
# Preprocess.py creates the feature df (i.e., input_df)
# submission.py uses the feature.csv , train GBR model, and creates the
# submission.csv
# **************************************************************************** #

# **************************************************************************** #
# Read in naturalized streamflow (target variable)
# **************************************************************************** #
metadata = pd.read_csv('metadata_TdPVeJC.csv')
# Train set
file_path = 'train_1128update.csv'
df = pd.read_csv(file_path)
df = df.rename(columns={'year': 'WY'})
df['site_id_short'] = df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)

# **************************************************************************** #
# Sites and forecast date
# **************************************************************************** #
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
# Read in and process predictors [PPT_acc, SWE_acc, Tmax_mean, Tmin_mean]
# **************************************************************************** #

# Write the prediction for the current issue date
pred_df_GBR_dict = {}

Data_folder = 'PRISM_USBR_forecast_Rodeo/Updated'
for date in submission_issue_date:
    print('**** forecastdate: ', date)
    random_state = 42
    quantiles = [0.1, 0.5, 0.9]
    cols = ['PPT_acc', 'SWE_acc', 'Tmax_mean', 'Tmin_mean', 'Tmax_var', 'Tmin_var', 'NSF_past3', 'NSF_past2', 'NSF_past1', 'drainage_area']
    features = pd.read_csv('preprocessed_dir/features_USGSreplace.csv')
    target = pd.read_csv('preprocessed_dir/target.csv')
    features['month_day'] = features['WY'].str[5:]
    features['year'] = features['WY'].str[:4].astype(int)
    merged_df = pd.merge(features, target, on=['WY', 'site_id'])
    result_df = merged_df[merged_df['month_day'] == date].copy()

    X = result_df[cols]
    y = result_df['volume']

    X_train = X
    y_train = y

    test_df = features[features['year'].isin(submission_test_year)]
    X_test = test_df[cols]

    # **************************************************************************** #
    # Gradient boost (for the current forecast date)
    # **************************************************************************** #
    # Later, you can load the parameters from the JSON file
    hyperparameters = {'max_depth': 10,
                       'max_features': 'sqrt',
                       'min_samples_leaf': 25,
                       'min_samples_split':25,
                       'n_estimators': 300,
                       'learning_rate': 0.05}
    # Perform quantile regression for each quantile
    GB_regressor_models = {}
    df_temps = {}
    for quantile in quantiles:
        GB_regressor = GradientBoostingRegressor(**hyperparameters, loss='quantile', alpha=quantile,
                                                 random_state=random_state)

        GB_regressor.fit(X_train, y_train)
        dump(GB_regressor, 'trained_models/'+date+"_"+str(quantile)+"_model.dat")

        df_temp = pd.DataFrame(columns=['site_id', 'WY', 'value'])
        df_temp['site_id'] = test_df['site_id']
        df_temp['WY'] = test_df['WY']
        df_temp['value'] = GB_regressor.predict(X_test)
        GB_regressor_models[quantile] = GB_regressor
        df_temps[quantile] = df_temp
    predictions_GBR = {q: df for q, df in df_temps.items()}
    pred_df_GBR_dict[date] = predictions_GBR

# Assign values to the submission file
submission_GBR = submission_file.copy()

for site_id in submission_site_id:
    for date in submission_issue_date:
        for WY in submission_test_year:
            issue_date = str(WY) + '-' + str(date)
            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.1)
            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.5)
            Utils.update_smoke_submission(submission_GBR, pred_df_GBR_dict, date, site_id, WY, 0.9)

submission_GBR.to_csv('submission_GBR_USGSreplace.csv', index=False)

