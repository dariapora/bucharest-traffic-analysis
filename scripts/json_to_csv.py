import pandas as pd
import os
import glob
import json

input_folder = '../data/reports/json'
output_folder = '../data/reports/csv'
all_dataframes = []

time_map = {
    2: '06:00-08:00',
    3: '08:00-10:00',
    4: '10:00-12:00',
    5: '12:00-14:00',
    6: '14:00-16:00',
    7: '16:00-18:00',
    8: '18:00-20:00'
}

json_files = glob.glob(os.path.join(input_folder, '*.json'))

for filename in json_files:
    with open(filename, 'r') as f:
        data = json.load(f)

    df_expanded = pd.json_normalize(
        data['network']['segmentResults'],
        record_path=['segmentTimeResults'],
        meta=['streetName', 'distance', 'speedLimit', 'segmentId'],
        errors='ignore'
    )

    df_expanded['date'] = data['dateRanges'][0]['from']
    df_expanded['timeSet'] = df_expanded['timeSet'].map(time_map)

    ordered_columns = [
        'date',
        'timeSet',
        'streetName',
        'segmentId',
        'distance',
        'sampleSize',
        'speedLimit',
        'harmonicAverageSpeed',
        'medianSpeed',
        'averageSpeed',
        'standardDeviationSpeed',
        'travelTimeStandardDeviation',
        'averageTravelTime',
        'medianTravelTime',
        'travelTimeRatio',
        'speedPercentiles'
    ]

    df_reordered = df_expanded[ordered_columns]
    df_reordered = df_reordered.rename(columns={
        'timeSet': 'timeRange'
    })

    all_dataframes.append(df_reordered)

complete_df = pd.concat(all_dataframes, ignore_index=True)
complete_df.to_csv('../data/reports/csv/dataset.csv', index=False)