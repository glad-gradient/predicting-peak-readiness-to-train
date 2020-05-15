import numpy as np


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
