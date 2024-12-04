import sys
import subprocess
import os
import re
import numpy as np
import pandas as pd
import time
from sys import argv, path, executable, exit
import subprocess
from datetime import datetime, timezone
from packaging.version import Version, InvalidVersion
import xarray as xr

# Input directory to read test input from
input_dir = sys.argv[1]

# Output data directory to which to write predictions
output_dir = sys.argv[2]

submission_dir = sys.argv[3]

sys.path.append(output_dir)
sys.path.append(submission_dir)

from model import Model


def get_sla_array(files):
    """
    Read the NetCDF files and extract the 'sla' variable
    """
    # Read the NetCDF files
    ds = xr.open_dataset(files)

    # Extract the 'sla' variable
    X = ds['sla'].values

    # Close the dataset
    ds.close()

    return X


def get_prediction_data():

    # set test data and solution file
    # Read in the nc files in the input_dir as a list
    test_data_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    test_data_files.sort()

    # Read the NetCDF files and extract the 'sla' variable as a list of numpy arrays
    X_test = [get_sla_array(f) for f in test_data_files]


    return X_test


def install_from_whitelist(req_file):
    whitelist = open("/app/program/whitelist.txt", 'r').readlines()
    whitelist = [i.rstrip('\n') for i in whitelist]
    # print(whitelist)

    for package in open(req_file, 'r').readlines():
        package = package.rstrip('\n')
        package_version = package.split("==")
        if len(package_version) > 2:
            # invalid format, don't use
            print(f"requested package {package} has invalid format, will install latest version (of {package_version[0]}) if allowed")
            package = package_version[0]
        elif len(package_version) == 2:
            version_str = package_version[1]
            Version(version_str)
            # try:
            #     Version(version_str)
            # except InvalidVersion:
            #     print(f"requested package {package} has invalid version, will install latest version (of {package_version[0]}) if allowed")
            #     package = package_version[0]

        #print("accepted package name: ", package)
        #print("package name ", package_version[0])
        if package_version[0] in whitelist:
            # package must be in whitelist, so format check unnecessary
            subprocess.check_call([executable, "-m", "pip", "install", package])
            print(f"{package_version[0]} installed")
        else:
            exit(f"{package_version[0]} is not an allowed package. Please contact the organizers on GitHub to request acceptance of the package.")
# def tp_cut(predictions):

#     # answers file
#     test_data_file = os.path.join(input_dir, 'ligo_bb_50.npz')

#     # Read solutions
#     with np.load(test_data_file) as file:
#         y_test = file['ids']
#         predictions = (predictions >= np.percentile(predictions[y_test == np.ones(len(y_test))], 90)).astype(int)

#     return predictions


def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")


def save_prediction(prediction_prob):

    prediction_file = os.path.join(output_dir, 'test.predictions')

    predictions = np.array(prediction_prob)



    with open(prediction_file, 'w') as f:
        for ind, lbl in enumerate(predictions):
            str_label = ' '.join(map(str, lbl))  # Remove brackets, just space-separated numbers
            if ind < len(predictions) - 1:
                f.write(str_label + "\n")
            else:
                f.write(str_label)  # No newline for the last line


def main():
    """
     Run the pipeline
     > Load
     > Predict
     > Save
    """

    start = time.time()

    requirements_file = os.path.join(submission_dir, "requirements.txt")
    if os.path.isfile(requirements_file):
        install_from_whitelist(requirements_file)

    duration = time.time() - start
    print_pretty(f'Duration of the package installation: {duration}')

    start = time.time()

    print_pretty('Reading Data')
    X_tests = get_prediction_data()


    print_pretty('Starting Learning')
    m = Model()

    print_pretty('Making Prediction')
    # Loop over all events and predict if the events has an anomaly
    prediction_prob = [ m.predict(X_test) for X_test in X_tests ] 

    print_pretty('Saving Prediction')
    save_prediction(prediction_prob)

    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')


if __name__ == '__main__':
    main()
