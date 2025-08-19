import argparse
import os
import numpy as np
import pynbody
import datetime
from glob import glob


def read_snapshot(folder_path, log_file):
    s = pynbody.load(os.path.join(folder_path, 'snapdir_062', 'snap_062'))
    s.physical_units()  # use physical units
    header = s.properties

    # write properties in log file
    for key, value in header.items():
        log_file.write(f"{key}: {value}\n")

    log_file.write("\nFamilies:\n")

    for fam in s.families():
        particles = s[fam]  
        log_file.write(f"{fam}: {len(particles)} particles, 3rd root: {round(len(particles) ** (1/3))}\n")

    log_file.write("\nKeys:\n")

    for key in s.loadable_keys():
        log_file.write(f"{key}\n")#: {len(s[key])}\n")

    

    
    log_file.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log properties of simulation snapshots.')
    parser.add_argument('--folders_patterns', type=str, required=True, help='Regular expression for input folders.')
    parser.add_argument('--logs_folder', type=str, default='./logs', help='Folder to save logs.')

    args = parser.parse_args()



    # create logs folder if not exists
    os.makedirs(args.logs_folder, exist_ok=True)

    # create if not exists logs/log_properties.txt
    log_file_path = os.path.join(args.logs_folder, 'log_properties.txt')

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Script started at {datetime.datetime.now()}\n")
        log_file.write(f"folders patterns: {args.folders_patterns}\n")
        log_file.write(f"logs folder: {args.logs_folder}\n")

        # get properties
        folders = glob(args.folders_patterns)
        log_file.write(f"Found {len(folders)} folders.\n\n")
        for folder in folders:
            log_file.write(f"Getting properties for folder: {folder}\n")
            try:
                read_snapshot(folder, log_file)
            except Exception as e:
                log_file.write(f"Error getting properties for {folder}: {e}\n")

        log_file.write(f"Script finished at {datetime.datetime.now()}\n\n\n\n\n\n\n\n\n\n")


