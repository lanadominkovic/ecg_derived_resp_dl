from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback

import matplotlib.pyplot as plt

import wandb
wandb.login()

import os

from models.model import create_model
from utils import load_bidmc_data, sliding_window

def create_sweep_config():
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'kernel_size': {
                'values': [3, 5, 7, 27]
            },
            'reg': {
                'min': 1e-5,
                'max': 1e-2
            },
            'dropout': {
                'min': 0.1,
                'max': 0.6
            },
            'start_filters': {
                'values': [8, 16, 32]
            }
        }
    }


def visualize_predictions(set_name, model, windows_ecg, windows_resp):
    prediction = model.predict(windows_ecg)[0]
    windows_resp= tf.squeeze(windows_resp)

    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    ax[0].plot(windows_resp)
    ax[0].set_title(f'{set_name.capitalize()} Ground Truth')
    ax[1].plot(prediction)
    ax[1].set_title(f'{set_name.capitalize()} Prediction')

    wandb.log({f"{set_name}_predictions": wandb.Image(fig)}, commit=False)
    plt.close(fig)

def log_images(model, epoch, logs, fixed_sample_ecg_train, fixed_sample_resp_train, fixed_sample_ecg_valid, fixed_sample_resp_valid):
        wandb.log({"loss": logs['loss'], "correlation": logs['correlation'], "val_loss": logs['val_loss'], "val_correlation": logs['val_correlation']})
        #visualize_predictions('train', model, fixed_sample_ecg_train, fixed_sample_resp_train)
        #visualize_predictions('validation', model, fixed_sample_ecg_valid, fixed_sample_resp_valid)

def correlation(x, y): #todo: check this and see in papers what cross correlation is
    # Normalize y to the [0, 1] range
    min_y = tf.math.reduce_min(y)
    max_y = tf.math.reduce_max(y)
    r_up = tf.math.subtract(y, min_y)
    r_down = max_y - min_y
    new_y = r_up / r_down
    
    # Compute means
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(new_y)
    
    # Compute centered values
    xm, ym = x - mx, new_y - my
    
    # Compute correlation coefficient
    r_num = tf.reduce_sum(tf.multiply(xm, ym))
    r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm)), tf.reduce_sum(tf.square(ym))))
    r = r_num / r_den
    
    # Ensure the result is between -1 and 1
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    
    return 1 - r


def train():
    wandb.init()
    config = wandb.config

    sampling_rate = 125
    input_size_seconds = 16 # //2, *2
    downsampled_window_size = 1024 #? power of 2
    window_size = input_size_seconds * sampling_rate
    overlap = 0.25 #25%
    
    data, patients = load_bidmc_data()

    # k-folds for CV
    train_patients = []
    test_patients = []
    validation_patients = []
    k = 5
    kf = KFold(n_splits=k)
    train_ind = []
    test_ind = []

    for tr_ind, te_ind in kf.split(patients):
        train_ind.append(tr_ind)
        test_ind.append(te_ind)

    split_ind = 0
    train_index, test_index = train_ind[split_ind], test_ind[split_ind]
    test_index, validation_index = train_test_split(test_index, test_size=0.5, random_state=42)
    train_patients = [patients[i] for i in train_index]
    validation_patients = [patients[i] for i in validation_index]
    test_patients = [patients[i] for i in test_index]

    windows_ecg_train, windows_resp_train, windows_ecg_validation, windows_resp_validation, windows_ecg_test, windows_resp_test = sliding_window(data, window_size, downsampled_window_size, overlap, train_patients, validation_patients, test_patients)

    if tf.config.list_physical_devices('GPU'):
        print("GPU enabled")
        with tf.device('/GPU:0'):
            windows_ecg_train = tf.convert_to_tensor(windows_ecg_train, dtype=tf.float32)
            windows_resp_train = tf.convert_to_tensor(windows_resp_train, dtype=tf.float32)
            windows_ecg_validation = tf.convert_to_tensor(windows_ecg_validation, dtype=tf.float32)
            windows_resp_validation = tf.convert_to_tensor(windows_resp_validation, dtype=tf.float32)
            windows_ecg_test = tf.convert_to_tensor(windows_ecg_test, dtype=tf.float32)
            windows_resp_test = tf.convert_to_tensor(windows_resp_test, dtype=tf.float32)

    fixed_index = 5
    fixed_sample_ecg_train = tf.expand_dims(windows_ecg_train[fixed_index], axis=0)
    fixed_sample_resp_train = tf.expand_dims(windows_resp_train[fixed_index], axis=0)

    fixed_sample_ecg_valid = tf.expand_dims(windows_ecg_validation[fixed_index], axis=0)
    fixed_sample_resp_valid = tf.expand_dims(windows_resp_validation[fixed_index], axis=0)
    
    lr = wandb.config.learning_rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    model = create_model(config)
    
    # define callbacks
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, cooldown=5, mode='min', min_lr=1e-7)
    #filepath = os.path.join('Users/lanacaldarevic/ecg_derived_resp_dl/models', f'model_crossval{split_ind}-size{start_filters}-input{input_size}_weights-improvement.h5')
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=20)
    lambda_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: log_images(model, epoch, logs, fixed_sample_ecg_train, fixed_sample_resp_train, fixed_sample_ecg_valid, fixed_sample_resp_valid))

    callbacks = [wandb.keras.WandbCallback(), early_stopping]
    model.compile(loss='mse', metrics=[correlation], optimizer=optimizer)
    
    print("Model training starting")
    model.fit(windows_ecg_train, windows_resp_train,
              epochs=200,
              batch_size=wandb.config.batch_size,
              shuffle=True,
              callbacks=callbacks,
              validation_data=(windows_ecg_validation, windows_resp_validation))
    
    model.save(os.path.join('/models', f'combined_model{split_ind}-size{config.start_filters}-input{1024}.h5'))

    wandb.finish()


if __name__ == "__main__":
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="biosignal_deeplearning")

    # Run sweep
    wandb.agent(sweep_id, train)
