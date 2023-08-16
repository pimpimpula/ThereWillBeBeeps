import glob
import os
import pickle

import numpy as np
import pandas as pd
from pathlib import Path


def get_path(folder_name: str):
    """
    Args:
        folder_name: "data", "figures", "raw_data", "audiograms" or "dataframes"

    Returns:
        path: path to the requested folder
    """

    # Get the current directory (i.e., the directory the script is running in)
    current_dir = Path(os.getcwd())

    # Construct the path to the raw_data directory
    if folder_name in ["raw_data", "audiograms", "dataframes"]:
        path = current_dir.parent.parent / 'data' / folder_name
    elif folder_name in ["data", "figures"]:
        path = current_dir.parent.parent / folder_name
    else:
        raise FileNotFoundError("Can't recognize the folder requested!")

    return path


def translate_conditions(pred):
    """
    Translate pred to labels used in the article.

    Args:
        pred: 'both', 'time', 'frequency' or 'none'

    Returns:
        translated pred ('FT', 'T', 'F' or 'R')

    """
    dictionary = {"both": "FT",
                  "frequency": "F",
                  "time": "T",
                  "none": "R",
                  # Bit ugly but quick fix
                  "FT": "FT",
                  "F": "F",
                  "T": "T",
                  "R": "R"
                  }

    return dictionary[pred]


def exclude_participants(participants):
    """
    Exclude participants:

    - `tvzljm` lied about their age and did not finish the experiment
    - `lkbxgs` had data issues (no Bayesian audiogram saved, all the following experiments were based on the wrong audiogram)
    - `wquuex` had > 20 dB HL
    - `bihhjl` is the HL participant
    - `tyrfqt`, `ikieoz`, `gtyzck` and `ttuwra` did a former version of the 3-AFC task

    """

    ignore_participants = ['tvzljm', 'lkbxgs', 'wquuex', 'bihhjl', 'tyrfqt', 'ikieoz', 'gtyzck', 'ttuwra', '.DS_Store']

    return [participant for participant in participants if participant not in ignore_participants]


def filter_files_by_suffix(participant, paradigm, suffix):
    path = os.path.join(get_path('raw_data'), participant, paradigm)
    csv_files = list(filter(os.path.isfile, glob.glob(os.path.join(path, suffix))))

    # sort by most recent (by reverse alphabetical order based on date in filename)
    csv_files.sort(key=lambda x: x, reverse=True)

    return csv_files


def fetch_paradigm_raw_data(participant, paradigm,
                            correct_continuous=False, filter_rts=False, use_all_available_data=False):
    """
    Gets raw data for one participant, for a chosen task.

    Args:
        participant (str): Participant ID.
        paradigm  (str): Task name
        correct_continuous (bool): Decide whether to correct responses from the Continuous task
            (shift by 1 trial, filter RTs < 1sec)
        use_all_available_data (bool): Decide whether to include data from first attempts at tasks if applicable


    Returns:
        DataFrame: The participant's data compiled from all CSV files.
    """

    csv_files = filter_files_by_suffix(participant, paradigm,
                                       "*.csv" if paradigm == 'Bayesian' else "*_1.csv")

    participant_data = pd.DataFrame()

    for file in csv_files:

        print("\n  -  Processing:", file.split('/')[-1])

        data = pd.read_csv(file)

        # Make sure all experiments use the same labels
        data.rename(columns={'Volume': 'Level',
                             'Catch trial': 'isCatchTrial',
                             'feedback.keys': 'responses',
                             'expName': 'paradigm',
                             'Prediction': 'pred'
                             }, inplace=True)

        first_file = False
        if paradigm != 'Bayesian':
            # Sometimes the Continuous exp crashed for some unknown reason, and had to be restarted.
            # This also happened once for the Cluster task (participant 'eqdcwr'
            # If that's the case, 'Resume previous experiment' == 1 for this file
            # Find whether the current file was the first one of the experiment, if so, don't look
            # at the following files. If it was a restarted experiment, check out the next files
            # until finding the first one.
            first_file = True if data['Resume previous experiment'].unique()[0] == 0 else False

            print('   *** New exp file ***' if first_file else "   *** Resumed exp file***")
            print('-----')

        # Check the columns in the csv file
        try:
            if paradigm in 'Continuous':
                data = data[['sweeps.thisN', 'trials.thisN', 'pred', 'Frequency', 'Level',
                             'isCatchTrial', 'responses', 'feedback.started', 'feedback.rt',
                             'participant', 'Resume previous experiment', 'paradigm']]
            elif paradigm == 'Cluster':
                data = data[['trials.thisN', 'pred', 'Frequency', 'Level',
                             'isCatchTrial', 'responses', 'feedback.rt',
                             'participant', 'Resume previous experiment', 'paradigm']]
            elif paradigm == 'Bayesian':
                data = data[['trials.thisN', 'Frequency', 'Level',
                             'isCatchTrial', 'responses', 'feedback.rt',
                             'participant', 'paradigm']]
            else:
                print("Paradigm not recognized:", paradigm)

        except KeyError:
            # Fix case of 'ofgjwt' 'ddmfvc' and 'nfsmrp' where there no 'feedback.rt' column was created in the first file
            # (because they didn't detect any of the tones presented)
            if participant in ['ofgjwt', 'ddmfvc', 'nfsmrp'] and paradigm == 'Continuous':
                data = data[['sweeps.thisN', 'trials.thisN', 'pred', 'Frequency', 'Level',
                             'isCatchTrial', 'responses', 'feedback.started',
                             'participant', 'Resume previous experiment', 'paradigm']]
                data.insert(9, 'feedback.rt', "[]")
            else:
                print(
                    '\x1b[0;31;40m' + f"{KeyError}: {participant.upper()}, - one file with wrong columns discarded:" + '\x1b[0m',
                    file)
                print(data.columns)
                continue

        # Correct Continuous responses for this file
        if correct_continuous:
            from scripts.preprocessing.funcs.fix_continuous_responses import correct_data
            data = correct_data(data, participant, filter_rts)

        participant_data = pd.concat([participant_data, data])

        # If this was the first file for a session, stop looking at the rest of the files
        # THIS IS DIFFERENT FROM WHAT WAS DONE PREVIOUSLY, WHERE I USED ALL OF THE AVAILABLE DATA
        if (first_file or paradigm == 'Bayesian') and not use_all_available_data:
            break

    return participant_data


def fetch_initialization_data(participant):
    """
    Fetch the initialization phase data for a participant.

    Args:
        participant (str): Participant ID.

    Returns:
        dict: The initialization data for the participant. None if no data is found.
    """

    init_files = filter_files_by_suffix(participant, 'Bayesian', "*_init.pkl")

    if init_files:
        with (open(init_files[-1], "rb")) as f_init:
            init_data = pickle.load(f_init)
    else:
        print(f"No initialization data found for participant {participant}...")
        init_data = None

    return init_data


def fetch_audiogram_data(participant, paradigm, pred=None, init=False):
    """
    Fetch the most recent audiogram data for a participant.

    Args:
        participant (str): Participant's identifier.

    Returns:
        dict: The audiogram data for the participant. None if no data is found.
    """

    # Filter based on the paradigm
    if paradigm == 'Bayesian':
        tag = "*_init.pkl" if init else "*_audiogram.pkl"
    elif paradigm == 'Continuous':
        tag = f"*{pred}_fixed.pkl"
    elif paradigm == 'Cluster':
        tag = f"*{pred}_audiogram.pkl"
    else:
        raise ValueError(f"Paradigm {paradigm} not recognized.")

    pkl_files = filter_files_by_suffix(participant, paradigm, tag)

    if pkl_files:
        # print(f"Opening audiogram for {participant}: ({paradigm if pred is None else f'{paradigm}/{pred}'})...")
        # print(pkl_files[-1])

        with open(pkl_files[-1], "rb") as f:
            participant_data = pickle.load(f)

            # Remove extra response in Bayesian paradigm?
            if paradigm == 'Bayesian' and init is None:
                participant_data['y'] = participant_data['y'][:-1]
    else:
        print(
            f"No audiogram found for participant {participant} ({paradigm if pred is None else f'{paradigm}/{pred}'})...")
        participant_data = None

    return participant_data


def set_audiograms_directory(aud_path, paradigm):
    aud_dir = os.path.join(aud_path, paradigm)
    if not os.path.exists(aud_dir):
        print(f"Created folder {aud_dir}")
        os.makedirs(aud_dir)
    return aud_dir


def load_gmsi_results():
    df = pd.read_excel(os.path.join(get_path('dataframes'), "gms_scoring.xlsx"),
                       usecols="AR,BF", header=0)  # "AR,BA:BF" for all components
    df = df.loc[df['ID'] != 0]
    df.rename(columns={'ID': 'participant', 'FG (General Sophistication)': 'gmsi'}, inplace=True)
    # df['Participant No'] = df['Participant No'].astype(int)
    return df


def load_age_info():
    df = pd.read_excel(os.path.join(get_path('dataframes'), "participant_handler.xlsx"),
                       usecols="C,D", header=0, skiprows=[38, 39, 40, 41, 42])
    df = df.loc[~df['ID'].isna()]
    df.rename(columns={'ID': 'participant', 'Age': 'age'}, inplace=True)
    # df['Participant No'] = df['Participant No'].astype(int)
    return df


def interp(x, x_axis, y_values):
    """
    Performs linear interpolation
    """

    xdist = np.array(x_axis) - x

    # look for the closest values to x in x_axis
    valueup = np.min(xdist[xdist > 0])
    indxup = np.where(xdist == valueup)[0][0]
    x_up = x_axis[indxup]

    valuedown = np.max(xdist[xdist < 0])
    indxdown = np.where(xdist == valuedown)[0][0]
    x_down = x_axis[indxdown]

    # interpolate y
    y = (x - x_down) / (x_up - x_down) * y_values[indxup] + \
        (x_up - x) / (x_up - x_down) * y_values[indxdown]

    return y
