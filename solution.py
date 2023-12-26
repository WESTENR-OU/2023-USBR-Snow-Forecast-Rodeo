"""This is the final submitted solution."""

from collections.abc import Hashable
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import warnings
import os
from joblib import load
from datetime import timedelta

warnings.filterwarnings('ignore')


def read_predictors(file_path, site_id_short, metadata_path, predictor_is_SWE=False, predictor_is_NSF=False):
    '''
    :param file_path: path to the predictor csv time series
    :param site_id_short:
    :return: A dataframe ['site_id'(full name), Date, Value]
    '''
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Rename columns
    if predictor_is_SWE:
        df = df.drop(columns=['Average Accumulated Water Year PPT (in)'])
        df.columns = ['Date', 'Value']
        df['Value'] = df['Value'] * 25.4  # convert inch to mm
    elif predictor_is_NSF:
        site_id_full = get_site_full_id(site_id_short, metadata_path)
        df.columns = ['site_id', 'WY', 'year', 'month', 'Value']
        # Check if site_id_full is in the DataFrame
        if site_id_full in df['site_id'].values:
            # Filter the DataFrame for the specific site_id_full
            df = df[df['site_id'] == site_id_full]
            df = df.drop(columns=['site_id'])
        else:
            water_years = list(range(1982, 2024))
            # water_years = pd.date_range(start='1981-01-01', end='2023-12-31', freq='D')
            # If site_id_full is not found, create a new DataFrame with the specified site_id
            new_data = {'site_id': [site_id_full] * len(water_years), 'WY': water_years, 'year': np.nan,
                        'month': np.nan,
                        'Value': np.nan}
            df = pd.DataFrame(new_data)
            return df
    else:
        df.columns = ['Date', 'Value']
    df.insert(0, 'site_id', get_site_full_id(site_id_short, metadata_path))

    return df


def get_site_full_id(site_id_short: str, metadata_path: str):
    df = pd.read_csv(metadata_path)
    site_id_dict = {}
    for i in range(len(df['site_id'])):
        short_id = df['site_id_short'][i]
        full_id = df['site_id'][i]
        site_id_dict[short_id] = full_id
    site_full_id = site_id_dict.get(site_id_short, None)
    return site_full_id

def get_site_short_id(site_id_full: str, metadata_path: str):
    df = pd.read_csv(metadata_path)
    site_id_dict = {}
    for i in range(len(df['site_id'])):
        short_id = df['site_id_short'][i]
        full_id = df['site_id'][i]
        site_id_dict[full_id] = short_id
    site_short_id = site_id_dict.get(site_id_full, None)  # Use site_id_full as the key
    return site_short_id

def slice_df(df: pd.DataFrame,
             start_year: int,
             end_year: int):
    condition = (df['WY'] >= start_year) & (df['WY'] <= end_year)
    return df[condition]


def calculate_mean(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
    subset = df.loc[mask]
    return subset['Value'].mean()


def compute_var(df: pd.DataFrame,
                forcast_date: str
                ):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    grouped_df = df.groupby('WY')
    var_df = pd.DataFrame(columns=['site_id', 'WY', 'Var'])
    for WY, group in grouped_df:
        date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
        group['Date'] = pd.to_datetime(group['Date'], format='%Y-%m-%d')
        group_before_forecast = group[group['Date'] < date]
        site_id = df[df['Date'] == date]['site_id'].values[0]
        var = group_before_forecast['Value'].var()
        var_df = pd.concat([var_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Var': [var]})])
    return var_df


def compute_monthly_mean(df: pd.DataFrame,
                         forecast_date: str,
                         swe: bool = False
                         ):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1

    mean_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean_past3', 'Mean_past2', 'Mean_past1'])
    WYs = list(range(1982, 2023 + 1))
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        # Calculate the dates 90, 60, and 30 days before the forecast_date
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        date_90before = date - timedelta(days=90)
        date_60before = date - timedelta(days=60)
        date_30before = date - timedelta(days=30)

        # Calculate means for the specified date ranges
        mean_90to60 = calculate_mean(df_WY, date_90before, date_60before)
        mean_60to30 = calculate_mean(df_WY, date_60before, date_30before)
        mean_30tocurrent = calculate_mean(df_WY, date_30before, date)

        # Update the DataFrame with the calculated means
        site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
        mean_df = pd.concat([mean_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY],
                                                    'Mean_past3': [mean_90to60],
                                                    'Mean_past2': [mean_60to30],
                                                    'Mean_past1': [mean_30tocurrent]})])
    return mean_df


def compute_acc(df: pd.DataFrame,
                forcast_date: str,
                saving_path: str,
                mode: str = 'acc',
                swe: bool = False
                ):
    '''
    :param df: Predictors time series data frame with
    1st col: "site_id"
    2nd col: "Date" ['%m-%d']. Do not specify the year.
    3rd col: "Value"
    :param forcast_date: str ['mm-dd'], e.g. "01-01","01-15"...
    :param saving_path: path to save the accumulated value df to csv
    :param mode: str ['acc','mean']. Default: 'acc'
    mode = 'mean' to compute past mean values (designed for Tmax and Tmin)
    :return: accumualted value until the forecast date in the same WY
    '''

    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    WYs = None
    if swe:
        WYs = df.WY.unique()
    if not swe:
        WYs = df.WY.unique()[1:]
    if mode == 'mean':
        # Accumulate the past values in the same WY
        df['Mean'] = df.groupby('WY')['Value'].cumsum()
        mean_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean'])
        for WY in WYs:  # remove WY 1981 since the record is incomplete
            date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
            date_diff = (date - pd.to_datetime(str(WY) + '-' + '10-01', format='%Y-%m-%d')).days
            site_id = df[df['Date'] == date]['site_id'].values[0]
            mean = df[df['Date'] == date]['Mean'].values[0] / date_diff
            mean_df = pd.concat([mean_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Mean': [mean]})])
        return df, mean_df
    # Accumulate the past values in the same WY
    df['Acc'] = df.groupby('WY')['Value'].cumsum()
    acc_df = pd.DataFrame(columns=['site_id', 'WY', 'Acc'])
    for WY in WYs:  # remove WY 1981 since the record is incomplete
        date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
        site_id = df[df['Date'] == date]['site_id'].values[0]
        acc = df[df['Date'] == date]['Acc'].values[0]
        acc_df = pd.concat([acc_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Acc': [acc]})])
    return df, acc_df


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    metadata_path = os.path.join(src_dir, 'data/metadata_TdPVeJC.csv')
    metadata = pd.read_csv(metadata_path)
    # Train set
    target_tr_path = os.path.join(src_dir, 'data/train_1128update.csv')
    target_df = pd.read_csv(target_tr_path)
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
    monthly_NSF_tr_path = os.path.join(src_dir, 'data/train_monthly_naturalized_flow_1128update.csv')
    monthly_NSF_test_path = os.path.join(src_dir, 'data/test_monthly_naturalized_flow_1128update.csv')
    site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
    site_id_short_in_monthlyNSF = [get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
    site_id_short_not_in_monthlyNSF = list(set(target_df['site_id_short'].unique()) - set(site_id_short_in_monthlyNSF))
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

    # **************************************************************************** #
    # Read in and process predictors [PPT_acc, SWE_acc, Tmax_mean, Tmin_mean]
    # **************************************************************************** #
    # Write the prediction for the current issue date
    Data_folder = os.path.join(src_dir, 'data/Updated')

    concatenated_PPT = []
    concatenated_SWE = []
    concatenated_Tmax = []
    concatenated_Tmin = []
    concatenated_Tmax_var = []
    concatenated_Tmin_var = []
    concatenated_NSF = []
    concatenated_monthly_NSF = []
    concatenated_drainage_area = []

    model_dict = {}
    for date in forecast_date:
        print(f'forecast date: {date}')
        model_10_path = os.path.join(os.path.join(src_dir, 'trained_models'), date + "_" + str(0.1) + "_model.dat")
        model_50_path = os.path.join(os.path.join(src_dir, 'trained_models'), date + "_" + str(0.5) + "_model.dat")
        model_90_path = os.path.join(os.path.join(src_dir, 'trained_models'), date + "_" + str(0.9) + "_model.dat")
        pred_model_10 = load(model_10_path)
        pred_model_50 = load(model_50_path)
        pred_model_90 = load(model_90_path)

        model_dict[date] = [pred_model_10, pred_model_50, pred_model_90]

        if date[:2] == '07':
            site_id_short = site_id_short[site_id_short != 'DLI']
        for site in site_id_short:
            # Predictors
            SWE_folder_path = os.path.join(Data_folder, 'SWE_UA')
            PPT_folder_path = os.path.join(Data_folder, 'PPT')
            Tmax_folder_path = os.path.join(Data_folder, 'T_max')
            Tmin_folder_path = os.path.join(Data_folder, 'T_min')

            SWE_df = read_predictors(SWE_folder_path + '/%s.csv' % site, site, metadata_path, predictor_is_SWE=True)
            PPT_df = read_predictors(PPT_folder_path + '/%s.csv' % site, site, metadata_path)
            Tmax_df = read_predictors(Tmax_folder_path + '/%s_Tmax.csv' % site, site, metadata_path)
            Tmin_df = read_predictors(Tmin_folder_path + '/%s_Tmin.csv' % site, site, metadata_path)
            NSF_train_df = read_predictors(monthly_NSF_tr_path, site, metadata_path, predictor_is_NSF=True)
            NSF_test_df = read_predictors(monthly_NSF_test_path, site, metadata_path, predictor_is_NSF=True)

            PPT_df, PPT_acc_df = compute_acc(PPT_df, date, PPT_folder_path + '/%s_%s_acc.csv' % (site, date))
            SWE_df, SWE_acc_df = compute_acc(SWE_df, date, SWE_folder_path + '/%s_%s_acc.csv' % (site, date),
                                             swe=True)
            Tmax_df, Tmax_mean_df = compute_acc(Tmax_df, date,
                                                Tmax_folder_path + '/%s_%s_mean.csv' % (site, date),
                                                mode='mean')
            Tmin_df, Tmin_mean_df = compute_acc(Tmin_df, date,
                                                Tmin_folder_path + '/%s_%s_mean.csv' % (site, date),
                                                mode='mean')
            Tmax_var_df = compute_var(Tmax_df, date)
            Tmin_var_df = compute_var(Tmin_df, date)

            # Slice both predictors and predictant to desired period (defined earlier)
            NSF_df = slice_df(df_by_sites[site], start_year, end_year)
            PPT_df_tr = slice_df(PPT_acc_df, start_year, end_year).rename(columns={'Acc': 'PPT_acc'})
            SWE_df_tr = slice_df(SWE_acc_df, start_year, end_year).rename(columns={'Acc': 'SWE_acc'})
            Tmax_df_tr = slice_df(Tmax_mean_df, start_year, end_year).rename(columns={'Mean': 'Tmax_mean'})
            Tmin_df_tr = slice_df(Tmin_mean_df, start_year, end_year).rename(columns={'Mean': 'Tmin_mean'})
            Tmax_var_df_tr = slice_df(Tmax_var_df, start_year, end_year).rename(columns={'Var': 'Tmax_var'})
            Tmin_var_df_tr = slice_df(Tmin_var_df, start_year, end_year).rename(columns={'Var': 'Tmin_var'})
            NSF_train_df = slice_df(NSF_train_df, start_year, end_year)
            NSF_test_df = slice_df(NSF_test_df, start_year, end_year)

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
                arr_interp = np.interp(np.arange(len(NSF_train_values_tab)), non_nan_indices,
                                       NSF_train_values_tab[non_nan_indices])
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
                arr_interp = np.interp(np.arange(len(NSF_test_values_tab)), non_nan_indices,
                                       NSF_test_values_tab[non_nan_indices])
                NSF_test_values_tab = arr_interp.reshape(10, 3)
            else:
                NSF_test_values_tab = np.full((10, 3), np.nan)

            NSF_df_tr = pd.DataFrame(NSF_train_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
            NSF_df_tr.insert(0, 'site_id', get_site_full_id(site, metadata_path))
            WY_tr = np.array([1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
                              1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                              2004, 2006, 2008, 2010, 2012, 2014,
                              2016, 2018, 2020, 2022])
            NSF_df_tr.insert(1, 'WY', WY_tr)
            NSF_df_tr['WY'] = pd.to_datetime(NSF_df_tr['WY'].astype(str) + '-' + date)

            NSF_df_test = pd.DataFrame(NSF_test_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
            NSF_df_test.insert(0, 'site_id', get_site_full_id(site, metadata_path))
            WY_test = np.array([2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023])
            NSF_df_test.insert(1, 'WY', WY_test)
            NSF_df_test['WY'] = pd.to_datetime(NSF_df_test['WY'].astype(str) + '-' + date)
            monthly_NSF_df_tr = pd.concat([NSF_df_tr, NSF_df_test], ignore_index=True).sort_values(by='WY')
            if site in site_id_short_not_in_monthlyNSF:
                USGS_path = os.path.join(src_dir, 'data/%s.csv'% site)
                USGS_df = read_predictors(USGS_path, site, metadata_path)
                USGS_df['Date'] = pd.to_datetime(USGS_df['Date'], format='%m/%d/%Y')
                USGS_df['month_day'] = USGS_df['Date'].dt.strftime('%m-%d')
                # USGS_df = USGS_df[USGS_df['month_day'] == date].copy()
                USGS_mean_df = compute_monthly_mean(USGS_df, date)
                USGS_df_tr = slice_df(USGS_mean_df, start_year, end_year)
                monthly_NSF_df_tr['NSF_past3'] = USGS_df_tr['Mean_past3'].reset_index(drop=True)
                monthly_NSF_df_tr['NSF_past2'] = USGS_df_tr['Mean_past2'].reset_index(drop=True)
                monthly_NSF_df_tr['NSF_past1'] = USGS_df_tr['Mean_past1'].reset_index(drop=True)
            # get the drainage area of this site
            drainage_area = metadata[metadata['site_id'] == get_site_full_id(site, metadata_path)]['drainage_area']
            drainage_area_df_tr = PPT_df_tr.copy().rename(columns={'PPT_acc': 'drainage_area'})
            drainage_area_df_tr['drainage_area'] = drainage_area.values[0]

            concatenated_PPT.append(PPT_df_tr)
            concatenated_SWE.append(SWE_df_tr)
            concatenated_Tmax.append(Tmax_df_tr)
            concatenated_Tmin.append(Tmin_df_tr)
            concatenated_Tmax_var.append(Tmax_var_df_tr)
            concatenated_Tmin_var.append(Tmin_var_df_tr)
            concatenated_NSF.append(NSF_df)
            concatenated_monthly_NSF.append(monthly_NSF_df_tr)
            concatenated_drainage_area.append(drainage_area_df_tr)

    # Concatenate all DataFrames for different sites into one
    # Regional training
    PPT_df_tr = pd.concat(concatenated_PPT, ignore_index=True)
    SWE_df_tr = pd.concat(concatenated_SWE, ignore_index=True)
    Tmax_df_tr = pd.concat(concatenated_Tmax, ignore_index=True)
    Tmin_df_tr = pd.concat(concatenated_Tmin, ignore_index=True)
    Tmax_var_df_tr = pd.concat(concatenated_Tmax_var, ignore_index=True)
    Tmin_var_df_tr = pd.concat(concatenated_Tmin_var, ignore_index=True)
    monthly_NSF_df_tr = pd.concat(concatenated_monthly_NSF, ignore_index=True)
    drainage_area_df_tr = pd.concat(concatenated_drainage_area, ignore_index=True)

    target_NSF_df_tr = pd.concat(concatenated_NSF, ignore_index=True)

    input_df = pd.concat([
        PPT_df_tr,
        SWE_df_tr.drop(columns=['site_id', 'WY']),
        Tmax_df_tr.drop(columns=['site_id', 'WY']),
        Tmin_df_tr.drop(columns=['site_id', 'WY']),
        Tmax_var_df_tr.drop(columns=['site_id', 'WY']),
        Tmin_var_df_tr.drop(columns=['site_id', 'WY']),
        monthly_NSF_df_tr.drop(columns=['site_id', 'WY']),
        drainage_area_df_tr.drop(columns=['site_id', 'WY'])
    ],
        axis=1)
    return {'features': input_df, 'target': target_NSF_df_tr, 'models': model_dict}


def predict(
        site_id: str,
        issue_date: str,
        assets,
        src_dir: Path,
        data_dir: Path,
        preprocessed_dir: Path,
) -> tuple[float, float, float]:
    date = issue_date[-5:]
    input_df = assets['features']
    train_input = input_df[(input_df['WY'] == issue_date) & (input_df['site_id'] == site_id)].drop(
        columns=['WY', 'site_id'])
    pred_model_10 = assets['models'][date][0]
    pred_model_50 = assets['models'][date][1]
    pred_model_90 = assets['models'][date][2]

    pred10 = pred_model_10.predict(train_input)[0]
    pred50 = pred_model_50.predict(train_input)[0]
    pred90 = pred_model_90.predict(train_input)[0]

    return pred10, pred50, pred90
