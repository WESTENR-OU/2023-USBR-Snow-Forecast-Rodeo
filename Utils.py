import pdb

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Contains all the functions used in the scripts

def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    return np.maximum(quantile * errors, (quantile - 1) * errors)


def mean_quantile_loss(y_true, quantile_predictions, quantiles):
    losses = []

    for quantile in quantiles:
        predictions = quantile_predictions[quantile]['value']
        loss = np.mean(pinball_loss(y_true, predictions, quantile))
        losses.append(loss)

    mean_loss = np.mean(losses)
    return mean_loss


def read_predictors(file_path, site_id_short, predictor_is_SWE=False, predictor_is_NSF=False):
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
        site_id_full = get_site_full_id(site_id_short)
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
            new_data = {'site_id': [site_id_full] * len(water_years), 'WY': water_years, 'year': np.nan, 'month': np.nan,
                        'Value': np.nan}
            df = pd.DataFrame(new_data)
            return df
    else:
        df.columns = ['Date', 'Value']
    df.insert(0, 'site_id', get_site_full_id(site_id_short))

    return df


def get_site_full_id(site_id_short: str):
    df = pd.read_csv(
        "metadata_TdPVeJC.csv")
    site_id_dict = {}
    for i in range(len(df['site_id'])):
        short_id = df['site_id_short'][i]
        full_id = df['site_id'][i]
        site_id_dict[short_id] = full_id
    site_full_id = site_id_dict.get(site_id_short, None)
    return site_full_id


def get_site_short_id(site_id_full: str):
    df = pd.read_csv(
        "metadata_TdPVeJC.csv")
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
        mean_df.to_csv(saving_path, index=False)
        return df, mean_df
    # Accumulate the past values in the same WY
    df['Acc'] = df.groupby('WY')['Value'].cumsum()
    acc_df = pd.DataFrame(columns=['site_id', 'WY', 'Acc'])
    for WY in WYs:  # remove WY 1981 since the record is incomplete
        date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
        site_id = df[df['Date'] == date]['site_id'].values[0]
        acc = df[df['Date'] == date]['Acc'].values[0]
        acc_df = pd.concat([acc_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Acc': [acc]})])
    acc_df.to_csv(saving_path, index=False)
    return df, acc_df

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



def calculate_mean(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
    subset = df.loc[mask]
    return subset['Value'].mean()


def calculate_acc(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
    subset = df.loc[mask]
    return subset['Value'].sum()


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


def compute_acc_90days(df: pd.DataFrame,
                       forecast_date: str,
                       saving_path: str,
                       mode: str = 'acc',
                       swe: bool = False
                       ):
    '''
    :param df: Predictors time series data frame with
    1st col: "site_id"
    2nd col: "Date" ['%m-%d']. Do not specify the year.
    3rd col: "Value"
    :param forecast_date: str ['mm-dd'], e.g. "01-01","01-15"...
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
    # Assuming your input date is in the format 'mm-dd'
    # input_format = '%m-%d'
    # forecast_date = datetime.strptime(forecast_date, input_format)
    # Iterate over the DataFrame and calculate means for each date range
    if mode == 'acc':
        new_df = pd.DataFrame(columns=['site_id', 'WY', 'Acc'])
    elif mode == 'mean':
        new_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean'])
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        # Calculate the dates 90, 60, and 30 days before the forecast_date
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        date_90before = date - timedelta(days=90)

        # Calculate means for the specified date ranges
        mean_90tocurrent = calculate_mean(df_WY, date_90before, date)
        acc_90tocurrent = calculate_acc(df_WY, date_90before, date)

        # Update the DataFrame with the calculated means
        site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
        if mode == 'acc':
            new_df = pd.concat([new_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Acc': [acc_90tocurrent]})])
            # acc_df.to_csv(saving_path, index=False)
        if mode == 'mean':
            new_df = pd.concat([new_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Mean': [mean_90tocurrent]})])
            # mean_df.to_csv(saving_path, index=False)
    return df, new_df


def calculate_interval_coverage(true_values, lower_quantile_preds, upper_quantile_preds):
    """
    Calculate the interval coverage.

    Parameters:
    - true_values: Array of true values.
    - lower_quantile_preds: Array of predicted values for the lower quantile (e.g., 0.10).
    - upper_quantile_preds: Array of predicted values for the upper quantile (e.g., 0.90).

    Returns:
    - Interval coverage: Proportion of true values that fall within the predicted interval.
    """
    # Check if true values fall within the predicted interval
    within_interval = np.logical_and(true_values >= lower_quantile_preds, true_values <= upper_quantile_preds)

    # Calculate the proportion of true values within the interval
    coverage = np.mean(within_interval)

    return coverage


def update_smoke_submission(df, pred_df_dict, date, site_id, WY, quantile):
    issue_date = str(WY) + '-' + date
    pred_df = pred_df_dict[date][quantile].loc[
        (pred_df_dict[date][quantile]['site_id'] == site_id) & (pred_df_dict[date][quantile]['WY'] == issue_date)
        ]
    df.loc[
        (df['site_id'] == site_id) & (df['issue_date'] == issue_date),
        f'volume_{int(quantile * 100)}'
    ] = pred_df['value'].values
    return df
