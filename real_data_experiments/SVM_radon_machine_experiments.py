import argparse
import csv
import os
import sys
import time
import re
import numpy as np
import sklearn
from joblib import Parallel, delayed
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from diffprivlib.models import LogisticRegression as PrivateLogisticRegression

# Bit hacky.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from radon_machine.linear_base_radon_machine import RadonMachineLinearBase, AveragingMachineLinearBase


def orthogonal_vector(v):
    v[0:2, ] = v[1::-1, ]
    v[0] *= -1
    return v


def orthogonal_big_vector(v):
    res = v - v @ (np.random.rand(v.size)) * np.random.rand(v.size) / np.linalg.norm(v)

    return res * 10


def run_noisy_experiment(X_train, y_train, X_test, y_test, split_idx,
                         num_trials=10,
                         log_file=None,
                         max_height=10,
                         sigmas=None,
                         shuffle=True,
                         averaging=False,
                         base_estimator=LinearSVC, **kwargs):
    if sigmas is None:
        # sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]
        sigmas = [4]

    if averaging:
        radon_machine = AveragingMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                                   **kwargs)
    else:
        radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                               **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    num_outliers = 0

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        wrt = csv.writer(file)
        if split_idx == 0:
            wrt.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            radon_machine.sigma = sigma
            for trial in range(num_trials):
                radon_machine.set_estimators(original_estimators, shuffle)
                auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                wrt.writerow(
                    # [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                    [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, 0,
                     max(radon_machine.condition_numbers)])


def run_hypothesis_flipping_experiment(X_train, y_train, X_test, y_test, split_idx,
                                       num_trials=1,
                                       log_file=None,
                                       max_height=10,
                                       sigmas=None,
                                       outliers=None,
                                       shuffle=True,
                                       averaging=False,
                                       base_estimator=LinearSVC, **kwargs):
    # How far away to put outliers.
    outlier_multiplier = 10
    if averaging:
        radon_machine = AveragingMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                                   **kwargs)
    else:
        radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                               **kwargs)
    if outliers is None:  # is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
        outliers = [0, 50, 100, 150, 200]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]
        sigmas = [0.1, 0.2]
        sigmas = [0]

    # radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = -attacked_estimators[0, :]

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = outlier_multiplier*attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        # [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, 0,
                         max(radon_machine.condition_numbers)])


def run_hypothesis_90deg_experiment(X_train, y_train, X_test, y_test, split_idx,
                                    num_trials=10,
                                    log_file=None,
                                    max_height=10,
                                    sigmas=None,
                                    outliers=None,
                                    shuffle=True,
                                    base_estimator=LinearSVC, **kwargs):
    if outliers is None:  # is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = orthogonal_vector(attacked_estimators[0, :])

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])


def run_hypothesis_orthogonal_big_vec_experiment(X_train, y_train, X_test, y_test, split_idx,
                                                 num_trials=10,
                                                 log_file=None,
                                                 max_height=10,
                                                 sigmas=None,
                                                 outliers=None,
                                                 shuffle=True,
                                                 averaging=False,
                                                 base_estimator=LinearSVC, **kwargs):
    if averaging:
        radon_machine = AveragingMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                                   **kwargs)
    else:
        radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150,
                                               **kwargs)
    if outliers is None:  # is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
        outliers = [0, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]
        sigmas = [0.1]

    radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = orthogonal_big_vector(attacked_estimators[0, :])

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, 0,
                         max(radon_machine.condition_numbers)])


def run_hypothesis_flipping_big_vec_experiment(X_train, y_train, X_test, y_test, split_idx,
                                               num_trials=10,
                                               log_file=None,
                                               max_height=10,
                                               sigmas=None,
                                               outliers=None,
                                               shuffle=True,
                                               base_estimator=LinearSVC, **kwargs):
    if outliers is None:  # is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]
    np.random.seed(split_idx)
    radon_machine = RadonMachineLinearBase(base_estimator, 100, max_height, n_jobs=1, random_state=11905150 + split_idx,
                                           **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = -attacked_estimators[0, :] * 10

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])


def call_parallel(function, X, y, name, learner, **kwargs):
    log_file = f"results/{name}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}.csv"
    folds = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=11905150)

    with Parallel(10) as parallel:
        x = parallel(delayed(function)(
            X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx], i, log_file=log_file,
            base_estimator=learner, **kwargs)
                     for (i, (train_idx, test_idx)) in enumerate(folds.split(X)))


def preprocess_arff(arff_file):
    with open(arff_file, 'r') as f:
        lines = f.readlines()

    data_lines = lines[23:]
    # Remove {, }, and numbering
    data_lines = [re.sub(r'\{|}|[0-9] ', '', line) for line in data_lines]

    # Write preprocessed data to a new file
    with open('datasets/preprocessed_codrna.arff', 'w') as f:
        f.writelines(lines[:23])
        for line in data_lines:
            f.writelines([line[:-1] + ",1\n"])



def load_susy():
    """Load the SUSY dataset."""
    arr = np.load("datasets/SUSY.npy")
    X, y = arr[:, 1:], arr[:, 0][:]
    return X, y


def load_codrna():
    """Load the preprocessed codrna dataset."""
    data, _ = arff.loadarff('datasets/preprocessed_codrna.arff')
    arr = [(y == b'1', *x) for (y, *x) in data]
    arr = np.array(arr, dtype=np.float64)
    X, y = arr[:, 1:], arr[:, 0]
    return X, y


def load_sea():
    """Load the SEA dataset."""
    data, _ = arff.loadarff('datasets/SEA_50.arff')
    arr = [(a, b, c, d == b'groupA') for (a, b, c, d) in data]
    arr = np.array(arr, dtype=np.float64)
    X, y = arr[:, :-1], arr[:, -1]
    return X, y


def run_experiments(dataset_name):
    dataset_loaders = {
        "SUSY": load_susy,
        "codrna": load_codrna,
        "SEA": load_sea
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_loaders.keys())}")

    X, y = dataset_loaders[dataset_name]()

    # Ignore the last part of the data, to be used for privacy attacks.
    split_idx = int(0.8 * len(X))
    X = X[:split_idx]
    y = y[:split_idx]

    if dataset_name == "SUSY":
        call_parallel(run_noisy_experiment, X, y, "SUSY_SVM_noisy", LinearSVC)
        call_parallel(run_noisy_experiment, X, y, "SUSY_SVM_noisy_averaged", LinearSVC, averaging=True)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "SUSY_SVM_break", LinearSVC,
        #               sigmas=[0], outliers=range(50, 100), num_trials=100)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "SUSY_SVM_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "SUSY_SVM_big_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "SUSY_LogReg_flip", LogisticRegression)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "SUSY_LogReg_big_flip", LogisticRegression)
        pass

    elif dataset_name == "codrna":
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "codrnaNorm_SVM_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "codrnaNorm_SVM_big_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "codrnaNorm_LogReg_flip", LogisticRegression)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "codrnaNorm_LogReg_big_flip", LogisticRegression)
        pass

    elif dataset_name == "SEA":
        # call_parallel(run_noisy_experiment, X, y, "sea50_SVM_noisy", LinearSVC)
        # call_parallel(run_noisy_experiment, X, y, "sea50_SVM_noisy_averaged", LinearSVC, averaging=True)
        # call_parallel(run_hypothesis_orthogonal_big_vec_experiment, X, y, "sea50_SVM_noisy", LinearSVC)
        # call_parallel(run_hypothesis_orthogonal_big_vec_experiment, X, y, "sea50_SVM_noisy_averaged", LinearSVC, averaging=True)
        call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_SVM_flip_rm", LinearSVC)
        call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_SVM_flip_avg", LinearSVC, averaging=True)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_SVM_flip_rm", PrivateLogisticRegression, epsilon=1)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_SVM_flip_avg", PrivateLogisticRegression, averaging=True, epsilon=1)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_SVM_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "sea50_SVM_big_flip", LinearSVC)
        # call_parallel(run_hypothesis_flipping_experiment, X, y, "sea50_LogReg_flip", LogisticRegression)
        # call_parallel(run_hypothesis_flipping_big_vec_experiment, X, y, "sea50_LogReg_big_flip", LogisticRegression)


def main():
    parser = argparse.ArgumentParser(description="Run experiments on different datasets.")
    parser.add_argument("--dataset", choices=["SUSY", "codrna", "SEA"], help="Select the dataset to use.")
    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset else "SEA"

    print(f"Running experiments on dataset: {dataset_name}")
    run_experiments(dataset_name)


if __name__ == "__main__":
    main()
