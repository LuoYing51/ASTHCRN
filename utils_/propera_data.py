import os
import numpy as np
import argparse
import torch
import configparser
import pandas as pd


def read_and_generate_dataset(data_seq, num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sample：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.8)
    split_line2 = int(len(all_samples) * 1)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line1: split_line2])]

    train_Xw = training_set[0]
    train_Xd = training_set[1]
    train_Y = training_set[2]
    val_Xw = validation_set[0]
    val_Xd = validation_set[1]
    val_Y = validation_set[2]
    test_Xw = testing_set[0]
    test_Xd = testing_set[1]
    test_Y = testing_set[2]

    # train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    # val_x = np.concatenate(validation_set[:-2], axis=-1)
    # test_x = np.concatenate(testing_set[:-2], axis=-1)
    #
    # train_target = training_set[-2]  # (B,N,T)
    # val_target = validation_set[-2]
    # test_target = testing_set[-2]
    #
    # train_timestamp = training_set[-1]  # (B,1)
    # val_timestamp = validation_set[-1]
    # test_timestamp = testing_set[-1]

    return train_Xw, train_Xd, train_Y, val_Xw, val_Xd, val_Y, test_Xw, test_Xd, test_Y


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict, :, 5:6]

    return week_sample, day_sample, hour_sample, target


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    if len(x_idx) != num_of_depend:
        return None
    return x_idx[::-1]


def preprocess_data(data, seq_len, pred_len):  # [?,35,6]

    X_week, X_day, Target = [], [], []

    for i in range(len(data) - seq_len - pred_len):
        x_w = data[i:i + seq_len]
        x_d = data[i + 7 * 24 - seq_len:i + 7 * 24]
        target = data[i + 7 * 24:i + 7 * 24 + pred_len, :, 5:6]
        X_week.append(x_w)
        X_day.append(x_d)
        Target.append(target)
    X_week = np.array(X_week).transpose(0, 2, 3, 1)
    X_day = np.array(X_day).transpose(0, 2, 3, 1)
    Target = np.array(Target).squeeze(3).transpose(0, 2, 1)

    return X_week, X_day, Target


def preprocess_data_day(data, seq_len, pred_len):  # [?,35,6]

    X_hour, Target = [], []

    for i in range(len(data) - seq_len - pred_len):
        x_h = data[i:i + seq_len:, :, :]
        target = data[i + seq_len:i + seq_len + pred_len, :, :]  # [24,35]

        X_hour.append(x_h)
        Target.append(target)

    X_hour = np.array(X_hour)
    Target = np.array(Target)

    return X_hour, Target
