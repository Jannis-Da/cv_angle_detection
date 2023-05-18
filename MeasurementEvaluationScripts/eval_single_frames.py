import pickle as pkl
import pandas as pd
import numpy as np


def main():
    pkl_file_path = '../MeasurementData/EinzelbilderEva/2023-05-17_20-05-26_EvalFrames.pkl'

    with open(pkl_file_path, 'rb') as file:
        pkl_data = pkl.load(file)

    log_df = pd.DataFrame(pkl_data)

    gt_data = read_gt_csv()
    gt_df = calc_gt_angles(gt_data)

    merged_df = pd.merge(log_df, gt_df, on='Time', how='inner')

    diff_df = calc_diff(merged_df)

    mae = calc_mae(diff_df)
    mse = calc_mse(diff_df)
    mape = calc_mape(diff_df)
    std_dev = calc_std_dev(diff_df)

    print(f"MAE Angle1 = {round(mae[0], 4)} rad")
    print(f"MAE Angle2 = {round(mae[1], 4)} rad")
    print(f"MSE Angle1 = {round(mse[0], 5)} rad^2")
    print(f"MSE Angle2 = {round(mse[1], 5)} rad^2")
    print(f"MAPE Angle1 = {round(mape[0], 4)} %")
    print(f"MAPE Angle2 = {round(mape[1], 4)} %")
    print(f"STD_DEV Angle1 = {round(std_dev[0], 4)} rad")
    print(f"STD_DEV Angle2 = {round(std_dev[1], 4)} rad")

    diff_df.to_csv('../MeasurementData/EinzelbilderEva/EvalOutput.csv', sep=';', index=False, decimal='.')
    diff_df.to_pickle('../MeasurementData/EinzelbilderEva/EvalOutput.pkl')


def norm_vector(vector):
    return vector / np.linalg.norm(vector)


def calc_angle(v1, v2):
    # Normalize vectors
    v1_u = norm_vector(v1)
    v2_u = norm_vector(v2)

    # Dot-product of normalized vectors, limited to values between -1 and 1, calculate angle with arc-cos
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def check_side(point_a, point_b, point_c):
    # Calculate determinant of 2x2-matrix built with the 3 points
    position = np.sign((point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (point_b[1] - point_a[1])
                       * (point_c[0] - point_a[0]))
    return position


def calc_gt_angles(df):

    df['A'] = df.apply(lambda row: np.array([row['A_x'], row['A_y']]), axis=1)
    df['B'] = df.apply(lambda row: np.array([row['B_x'], row['B_y']]), axis=1)
    df['C'] = df.apply(lambda row: np.array([row['C_x'], row['C_y']]), axis=1)

    df = df.drop(['A_x', 'A_y', 'B_x', 'B_y', 'C_x', 'C_y'], axis=1)
    df['AB'] = df['B'] - df['A']
    df['BC'] = df['C'] - df['B']

    angle1 = np.zeros(len(df), float)
    angle2 = np.zeros(len(df), float)
    idx = 0

    ref_axis = df.iloc[0, 2] - df.iloc[0, 1]

    for index, row in df.iterrows():
        A = row['A']
        B = row['B']
        C = row ['C']
        AB = row['AB']
        BC = row['BC']
        if check_side(A, A + ref_axis, B) == (-1 | 0):
            angle1[idx] = calc_angle(ref_axis, AB)
        else:
            angle1[idx] = -calc_angle(ref_axis, AB)

        if check_side(B, B + AB, C) == (-1 | 0):
            angle2[idx] = calc_angle(AB, BC)
        else:
            angle2[idx] = -calc_angle(AB, BC)

        idx = idx+1

    df['gt_Angle1'] = angle1
    df['gt_Angle2'] = angle2

    return df


def calc_diff(df):
    df['abs_err_Angle1'] = abs(df['Angle1'] - df['gt_Angle1'])
    df['abs_err_Angle2'] = abs(df['Angle2'] - df['gt_Angle2'])
    return df


def calc_mae(df):
    mae_Angle1 = np.mean(df['abs_err_Angle1'])
    mae_Angle2 = np.mean(df['abs_err_Angle2'])
    return [mae_Angle1, mae_Angle2]


def calc_mse(df):
    mse_Angle1 = np.mean((df['gt_Angle1'] - df['Angle1'])**2)
    mse_Angle2 = np.mean((df['gt_Angle2'] - df['Angle2'])**2)
    return [mse_Angle1, mse_Angle2]


def calc_mape(df):
    mape_Angle1 = np.mean(np.abs((df['gt_Angle1'] - df['Angle1']) / df['gt_Angle1'])) * 100
    mape_Angle2 = np.mean(np.abs((df['gt_Angle2'] - df['Angle2']) / df['gt_Angle2'])) * 100
    return [mape_Angle1, mape_Angle2]


def calc_std_dev(df):
    std_dev_Angle1 = np.std(df['abs_err_Angle1'])
    std_dev_Angle2 = np.std(df['abs_err_Angle2'])
    return[std_dev_Angle1, std_dev_Angle2]


def read_gt_csv():
    gt_data_dir = '../MeasurementData/EinzelbilderEva/GroundTruthV1.csv'
    return pd.read_csv(gt_data_dir, delimiter=';', decimal='.', dtype=float, na_values='')


if __name__ == '__main__':
    main()