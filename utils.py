import os
from glob import glob

import numpy as np
import pandas as pd
from scipy import signal


def load_bidmc_data():
     # loading of bidmc
    path = "data/bidmc-ppg-and-respiration-dataset-1.0.0"
    EXT = "*Signals.csv"
    all_csv_files = [file for path, subdir, files in os.walk(path) for file in glob(os.path.join(path, EXT))]
    patients = []
    data = {}
    no_errors = 0
    for file in all_csv_files:
        try:
            df = pd.read_csv(file)
            X1, X2, X3, X4 = df[' PLETH'].values, df[' V'].values, df[' AVR'].values, df[' II'].values
            # X = np.concatenate([X1.reshape(len(X1),1),X2.reshape(len(X1),1),X3.reshape(len(X1),1),X4.reshape(len(X1),1)], axis=1)
            
            Y = df[' RESP'].values
            
            patient = int(file.split('/')[-1].split('_')[1])
            patients.append(patient)
            data[patient] = [X4, Y]
        except:
            no_errors += 1

    return data, patients


def sliding_window(data, window_size, downsampled_window_size, overlap, train_patients, validation_patients, test_patients):
    windows_ecg_train = []
    windows_resp_train = []

    for train_patient in train_patients:
    
        N = len(data[train_patient][0])
        max_step = int(N//(window_size*overlap))
        for step in range(max_step):
            ecg = data[train_patient][0][step * int(window_size*overlap):step * int(window_size*overlap) + window_size] 
            resp = data[train_patient][1][step * int(window_size*overlap):step * int(window_size*overlap) + window_size]
            
            if (ecg.min() < ecg.max()):
                normalized_ecg = (ecg-ecg.min())/(ecg.max()-ecg.min())-0.5
                #zero_centered_ecg = ecg - np.mean(ecg)
                #normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp-resp.min())/(resp.max()-resp.min())
                    #zero_centered_resp = resp - np.mean(resp)
                    #normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(normalized_resp, downsampled_window_size)
                    windows_ecg_train.append(np.float32(resampled_ecg))
                    windows_resp_train.append(np.float32(resampled_resp))
            
            
    windows_ecg_validation = []
    windows_resp_validation = []

    for validation_patient in validation_patients:
        N = len(data[validation_patient][0])
        max_step = int(N//(window_size*overlap))
        for step in range(max_step):
            ecg = data[validation_patient][0][step * int(window_size*overlap):step * int(window_size*overlap) + window_size] 
            resp = data[validation_patient][1][step * int(window_size*overlap):step * int(window_size*overlap) + window_size]
            
            if (ecg.min() < ecg.max()):
                normalized_ecg = (ecg-ecg.min())/(ecg.max()-ecg.min())-0.5
                #zero_centered_ecg = ecg - np.mean(ecg)
                #normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp-resp.min())/(resp.max()-resp.min())
                    #zero_centered_resp = resp - np.mean(resp)
                    #normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(normalized_resp, downsampled_window_size)
                    windows_ecg_validation.append(np.float32(resampled_ecg))
                    windows_resp_validation.append(np.float32(resampled_resp))
          
    windows_ecg_test = []
    windows_resp_test = []
    
    for test_patient in test_patients:
        N = len(data[test_patient][0])
        max_step = int(N//(window_size*overlap))
        for step in range(max_step):
            ecg = data[test_patient][0][step * int(window_size*overlap):step * int(window_size*overlap) + window_size] 
            resp = data[test_patient][1][step * int(window_size*overlap):step * int(window_size*overlap) + window_size]
            
            if (ecg.min() < ecg.max()):
                normalized_ecg = (ecg-ecg.min())/(ecg.max()-ecg.min())-0.5
                #zero_centered_ecg = ecg - np.mean(ecg)
                #normalized_ecg = zero_centered_ecg / np.std(zero_centered_ecg)
                resampled_ecg = signal.resample(normalized_ecg, downsampled_window_size)
                if resp.min() < resp.max():
                    normalized_resp = (resp-resp.min())/(resp.max()-resp.min())
                    #zero_centered_resp = resp - np.mean(resp)
                    #normalized_resp = zero_centered_resp / np.std(zero_centered_resp)
                    resampled_resp = signal.resample(normalized_resp, downsampled_window_size)
                    windows_ecg_test.append(np.float32(resampled_ecg))
                    windows_resp_test.append(np.float32(resampled_resp))

    windows_ecg_train = np.stack(windows_ecg_train, axis=0)
    windows_resp_train = np.stack(windows_resp_train, axis=0)
    windows_ecg_validation = np.stack(windows_ecg_validation, axis=0)
    windows_resp_validation = np.stack(windows_resp_validation, axis=0)
    windows_ecg_test = np.stack(windows_ecg_test, axis=0)
    windows_resp_test = np.stack(windows_resp_test, axis=0)

    windows_ecg_train = windows_ecg_train[:,:,np.newaxis]
    windows_resp_train = windows_resp_train[:,:,np.newaxis]
    windows_ecg_validation = windows_ecg_validation[:,:,np.newaxis]
    windows_resp_validation = windows_resp_validation[:,:,np.newaxis]
    windows_ecg_test = windows_ecg_test[:,:,np.newaxis]
    windows_resp_test = windows_resp_test[:,:,np.newaxis]

    return windows_ecg_train, windows_resp_train, windows_ecg_validation, windows_resp_validation, windows_ecg_test, windows_resp_test