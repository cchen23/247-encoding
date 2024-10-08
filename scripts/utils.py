import json
import os
import pickle
import psutil
from datetime import datetime


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return function_wrapper


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum


def write_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join(dictionary["full_output_dir"], "config.json")
    with open(config_file, "w") as outfile:
        outfile.write(json_object)


def print_profile():
    print("The CPU usage is: ", psutil.cpu_percent(4))
    print("RAM memory % used:", psutil.virtual_memory()[2])
    print("RAM Used (GB):", psutil.virtual_memory()[3]/1000000000)