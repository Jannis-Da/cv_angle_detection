import pandas as pd
import os

data = {'Time': [],
        'A_x': [],
        'A_y': [],
        'B_x': [],
        'B_y': [],
        'C_x': [],
        'C_y': []}

df = pd.DataFrame(data)

frames_dir = '../MeasurementData/EinzelbilderEva/2023-05-17_20-05-26_EvalFrames'
file_names = os.listdir(frames_dir)

ref_row = pd.Series(
    {'Time': 0.0, 'A_x': None, 'A_y': None, 'B_x': None, 'B_y': None, 'C_x': None, 'C_y': None})
df = pd.concat([df, ref_row.to_frame().T], ignore_index=True)

for file_name in file_names:
    start_index = file_name.find('_') + 1
    end_index = file_name.rfind('.')
    timestamp = float(file_name[start_index:end_index])
    new_row = pd.Series({'Time': timestamp, 'A_x': None,'A_y': None, 'B_x': None, 'B_y': None, 'C_x': None, 'C_y': None})
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

df = df.sort_values('Time')
df.to_csv('../MeasurementData/EinzelbilderEva/GroundTruth.csv', sep=';', index=False, decimal='.')

