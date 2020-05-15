import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from imblearn.over_sampling import SMOTE

from utils import split_sequences


with open('configs.json', 'r') as file:
    configs = json.load(file)

RANDOM_STATE = configs["RANDOM_STATE"]
MODEL_CHECKPOINT_PATH = configs["MODEL_CHECKPOINT_PATH"]
lookback_window = configs['LOOKBACK_WINDOW']
lookforward_window = configs['LOOKFORWARD_WINDOW']

df = pd.read_csv('dataset.csv')

X, y = split_sequences(
    df,
    lookback_window,
    lookforward_window,
    y_variable_name='readiness_group',
    exclude_columns=['pid', 'date', 'readiness_group']
)

nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx * ny))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
X_valid, _, y_valid, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_valid = minmax_scaler.transform(X_valid)

# balance training data
sm = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
X_train, y_train = sm.fit_sample(X_train, y_train)

nsamples = X_train.shape[0]
X_train = X_train.reshape((nsamples, nx, ny))

nsamples = X_valid.shape[0]
X_valid = X_valid.reshape((nsamples, nx, ny))

# Internal neural network parameters
input_dim = X_train.shape[-1]
output_dim = len(np.bincount(y))

input_shape = (lookback_window, input_dim)

batch_size = configs['CNN_MODEL']['BATCH_SIZE']
epochs = configs['CNN_MODEL']['EPOCHS']

input_layer = Input(input_shape)

conv1 = Conv1D(filters=512, kernel_size=4, padding='same')(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = Activation(activation='relu')(conv1)

gap_layer = GlobalAveragePooling1D()(conv1)

output_layer = Dense(output_dim, activation='softmax')(gap_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=configs['CNN_MODEL']['LEARNING_RATE']))

early_stop_callback = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)
path_checkpoint = f'{MODEL_CHECKPOINT_PATH}' \
    f'/readiness_cnn_classifier_keras_{lookback_window}_{lookforward_window}_{input_dim}_{output_dim}_{batch_size}'

checkpoint_callback = ModelCheckpoint(filepath=path_checkpoint + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

model.fit(
    x=X_train,
    y=y_train,
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    batch_size=batch_size,
    callbacks=[early_stop_callback, checkpoint_callback],
    verbose=1
)


