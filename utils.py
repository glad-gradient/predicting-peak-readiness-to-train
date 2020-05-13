from datetime import datetime
import logging
import os

import numpy as np
from pandas import DataFrame, read_csv, concat
import dateutil.parser


def get_data(list_subfolders_with_paths):
    srpe = DataFrame()
    wellness = DataFrame()
    injury = DataFrame()

    for i, path in enumerate(list_subfolders_with_paths):
        # read srpe data
        srpe_file = f'{path}\\pmsys\\srpe.csv'
        if os.path.exists(srpe_file):
            srpe_temp = read_csv(srpe_file)
            srpe_temp['pid'] = i + 1
            srpe_temp['end_date_time'] = srpe_temp['end_date_time'].apply(lambda x:
                                                                          datetime.utcfromtimestamp(int(
                                                                              dateutil.parser.parse(x).timestamp())))
            srpe_temp.sort_values('end_date_time', inplace=True)
            srpe = concat([srpe, srpe_temp], ignore_index=True)
        else:
            logging.warning(f"File {srpe_file} doesn't exist!")

        # read wellness data
        wellness_file = f'{path}\\pmsys\\wellness.csv'  # effective_time_frame
        if os.path.exists(wellness_file):
            wellness_temp = read_csv(wellness_file)
            wellness_temp['pid'] = i + 1
            wellness_temp['effective_time_frame'] = wellness_temp['effective_time_frame'].apply(
                lambda x: datetime.utcfromtimestamp(int(dateutil.parser.parse(x).timestamp()))
            )
            wellness_temp.sort_values('effective_time_frame', inplace=True)
            wellness = concat([wellness, wellness_temp], ignore_index=True)
        else:
            logging.warning(f"File {wellness_file} doesn't exist!")

        # read injury data
        injury_file = f'{path}\\pmsys\\injury.csv'
        if os.path.exists(injury_file):
            injury_temp = read_csv(injury_file)
            injury_temp['pid'] = i + 1
            injury_temp['effective_time_frame'] = injury_temp['effective_time_frame'].apply(
                lambda x: datetime.utcfromtimestamp(int(dateutil.parser.parse(x).timestamp()))
            )
            injury_temp.sort_values('effective_time_frame', inplace=True)
            injury = concat([injury, injury_temp], ignore_index=True)
        else:
            logging.warning(f"File '{injury_file}' doesn't exist!")

    return srpe, wellness, injury


def split_sequences(_df, n_steps_in, n_steps_out, y_variable_name, exclude_columns):
    _X, _y = list(), list()

    for _, data in _df.groupby('pid'):
        X_variables = data[data.columns[~data.columns.isin(exclude_columns)]].values
        y_variable = data[y_variable_name].values

        for i in range(len(y_variable)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix >= len(y_variable):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = X_variables[i:end_ix], y_variable[out_end_ix]
            _X.append(seq_x)
            _y.append(seq_y)
    return np.array(_X, dtype=np.float16), np.array(_y, dtype=int)
