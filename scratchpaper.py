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
pred_df_XGB_dict = {}
pred_df_mean_dict = {}


# TODO DELETE BELOW
mean_QL_df = pd.DataFrame(columns=['issue_date', 'GBR', 'XGB', 'Ensemble_Mean'])
IC_df = pd.DataFrame(columns=['issue_date', 'GBR', 'XGB', 'Ensemble_Mean'])
mean_QL_df['issue_date'] = forecast_date
IC_df['issue_date'] = forecast_date

# Write the prediction for the current issue date
pred_df_GBR = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_XGB = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_mean = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_GBR_dict = {}
pred_df_XGB_dict = {}
pred_df_mean_dict = {}
# TODO DELETE ABOVE


Data_folder = 'PRISM_USBR_forecast_Rodeo/Updated'
for date in submission_issue_date:
    print('**** forecastdate: ', date)
    random_state = 42
    quantiles = [0.1, 0.5, 0.9]
    cols = ['PPT_acc', 'SWE_acc', 'Tmax_mean', 'Tmin_mean', 'NSF_past3', 'NSF_past2', 'NSF_past1', 'drainage_area']
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

    test_df = result_df[result_df['year'].isin(submission_test_year)]
    X_test = test_df[cols]

    cols = ['PPT_acc', 'SWE_acc', 'Tmax_mean', 'Tmin_mean', 'Tmax_var', 'Tmin_var', 'NSF_past3', 'NSF_past2', 'NSF_past1',
            'drainage_area'
            ]  # TODO: ADD NSF FEATURES
    #cols = ['PPT_acc', 'SWE_acc', 'Tmax_mean', 'Tmin_mean', 'NSF_past3', 'NSF_past2', 'NSF_past1', 'drainage_area']
    X = result_df[cols]  # TODO: ADD NSF FEATURES
    y = result_df['volume']
    # Split the data into training and testing sets
    train_df = result_df[result_df['year'].isin(year_list[:28])]  # TODO DELETE IF NOT VAL OFFLINE
    X = train_df[cols]  # TODO DELETE IF NOT VAL OFFLINE
    y = train_df['volume']  # TODO DELETE IF NOT VAL OFFLINE
    # Split the data into training and testing sets
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0, random_state=random_state)
    X_train = X
    y_train = y
    test_df = features[features['year'].isin(submission_test_year)]
    X_test = test_df[cols]

    test_df = result_df[result_df['year'].isin(year_list[28:])]  # TODO DELETE IF NOT VAL OFFLINE
    X_test = test_df[cols]  # TODO DELETE IF NOT VAL OFFLINE
    y_test = test_df['volume']  # TODO DELETE IF NOT VAL OFFLINE
    site_id_test = test_df['site_id'].unique()  # TODO DELETE IF NOT VAL OFFLINE


    # **************************************************************************** #
    # Gradient boost (for the current forecast date)
    # **************************************************************************** #
    # Later, you can load the parameters from the JSON file
    # 67.89
    hyperparameters = {'max_depth': 10,
                       'max_features': 'sqrt',
                       'min_samples_leaf': 25,
                       'min_samples_split':25,
                       'n_estimators': 300,
                       'learning_rate': 0.05}

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
    predictions_XGB = {0.1: y_lower_df, 0.5: y_med_df, 0.9: y_upper_df}
    pred_df_XGB_dict[date] = predictions_XGB


    mean_QL = Utils.mean_quantile_loss(y_test, predictions_XGB, quantiles)
    IC = Utils.calculate_interval_coverage(y_test, predictions_XGB[quantiles[0]]["value"],
                                           predictions_XGB[quantiles[-1]]["value"])
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
# Assign values to the submission file
submission_GBR = submission_file.copy()
submission_XGB = submission_file.copy()
submission_mean = submission_file.copy()

for site_id in submission_site_id:
    for date in submission_issue_date:
        for WY in submission_test_year:
            issue_date = str(WY) + '-' + str(date)
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