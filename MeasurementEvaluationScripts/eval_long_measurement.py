import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    pkl_file_path = '../MeasurementData/LongEva/2023-05-18_17-23-12_EvalLong.pkl'

    with open(pkl_file_path, 'rb') as file:
        pkl_data = pkl.load(file)

    log_df = pd.DataFrame(pkl_data)

    missing_angle1_count = log_df['Angle1'].isna().sum()
    missing_angle1_prct = missing_angle1_count/len(log_df) * 100

    missing_angle2_count = log_df['Angle2'].isna().sum()
    missing_angle2_prct = missing_angle2_count/len(log_df) * 100

    print(f'Analysed Frames: {len(log_df)}')
    print(f'Not measured Angles for Arm 1: {round(missing_angle1_prct, 3)}%')
    print(f'Not measured Angles for Arm 2: {round(missing_angle2_prct, 3)}%')

    misdetection_df = log_df.drop(log_df[(log_df['Angle1'].notna()) & (log_df['Angle2'].notna())].index)
    misdetection_df.to_csv('../MeasurementData/LongEva/Misdetection.csv', sep=';', index=False, decimal='.')

    execution_time_df = log_df
    execution_time_df['ExecutionTime'] = execution_time_df['Time'].diff()
    execution_time_df = execution_time_df.drop(0)

    mean_execution_time = np.mean(execution_time_df['ExecutionTime'])
    std_dev_execution_time = np.std(execution_time_df['ExecutionTime'])
    median_execution_time = np.median(execution_time_df['ExecutionTime'])
    abs_deviations = np.abs(execution_time_df['ExecutionTime'] - median_execution_time)
    mad_execution_time = np.median(abs_deviations)
    print(f'Mean Execution Time: {round((mean_execution_time * 1000), 3)}ms')
    print(f'Standard Deviation Execution Time: {round((std_dev_execution_time * 1000), 3)}ms')
    print(f'Median Absolute Deviation Execution Time: {round((mad_execution_time * 1000), 3)}ms')


    execution_time_df.plot(x='Time', y=['ExecutionTime'], linestyle='-')
    plt.xlabel('Measurement Time [s]')
    plt.ylabel('Execution Time[s]')
    plt.title('Execution Time during Measurement ')
    plt.savefig('../MeasurementData/LongEva/Plots/ExecutionTimeEvalLong.pdf', format='pdf')
    #plt.show()


if __name__ == '__main__':
    main()