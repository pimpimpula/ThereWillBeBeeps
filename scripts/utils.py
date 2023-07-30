import glob
import os
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
                  "none": "R"
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

    exclude_participants = ['tvzljm', 'lkbxgs', 'wquuex', 'bihhjl', 'tyrfqt', 'ikieoz', 'gtyzck', 'ttuwra', '.DS_Store']

    return [participant for participant in participants if participant not in exclude_participants]


def filter_files_by_suffix(participant, paradigm, suffix):
    path = os.path.join(get_path('raw_data'), participant, paradigm)
    csv_files = list(filter(os.path.isfile, glob.glob(os.path.join(path, suffix))))

    # sort by most recent (by reverse alphabetical order based on date in filename)
    csv_files.sort(key=lambda x: x, reverse=True)

    return csv_files


def fetch_paradigm_raw_data(participant, paradigm, correct_continuous=False, use_all_available_data=False):
    """
    Gets raw data for one participant, for a chosen task.

    Args:
        participant (str): Participant's identifier.
        paradigm  (str):
        correct_continuous (bool): Decide whether to correct responses from the Continuous task (shift by 1 trial, filter RTs < 1sec)
        use_all_available_data (bool): Decide whether to include data from first attempts at tasks if applicable


    Returns:
        DataFrame: The participant's data compiled from all CSV files.
    """

    csv_files = filter_files_by_suffix(participant, paradigm,
                                       "*.csv" if paradigm == 'Bayesian' else "*_1.csv")

    participant_data = pd.DataFrame()

    for file in csv_files:

        print(file.split('/')[-1])

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

            print('New exp file\n' if first_file else "Resumed exp file\n")

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

            # Fix case of 'ofgjwt' where there no 'feedback.rt' column was created in the first file of Continuous
            # (because they didn't detect any of the tones presented)
            if participant == 'ofgjwt' and paradigm == 'Continuous':
                data = data[['sweeps.thisN', 'trials.thisN', 'pred', 'Frequency', 'Level',
                             'isCatchTrial', 'responses', 'feedback.started',
                             'participant', 'Resume previous experiment', 'paradigm']]
                data.loc['feedback.rt'] = "[]"
            else:
                print('\x1b[0;31;40m' + f"{KeyError}: {participant.upper()}, - one file with wrong columns discarded:" + '\x1b[0m', file)
                print(data.columns)
                continue

        # Correct Continuous responses for this file
        if correct_continuous:
            from scripts.preprocessing.funcs.fix_continuous_responses import correct_data
            data = correct_data(data, participant)

        participant_data = pd.concat([participant_data, data])

        # If this was the first file for a session, stop looking at the rest of the files
        # THIS IS DIFFERENT FROM WHAT WAS DONE PREVIOUSLY, WHERE I USED ALL OF THE AVAILABLE DATA
        if (first_file or paradigm == 'Bayesian') and not use_all_available_data:
            break


    return participant_data



def load_goldMSI_results():
    df = pd.read_excel(os.path.join(get_path('dataframes'), "gms_scoring.xlsx"),
                       usecols="AR,BF", header=0)  # "AR,BA:BF" for all components
    df = df.loc[df['ID'] != 0]
    df.rename(columns={'ID': 'participant', 'FG (General Sophistication)': 'gmsi'}, inplace=True)
    # df['Participant No'] = df['Participant No'].astype(int)
    return df


def load_age_info():
    df = pd.read_excel(os.path.join(get_path('dataframes'), "participant_handler.xlsx"),
                       usecols="C,D", header=0)
    df = df.loc[~df['ID'].isna()]
    df.rename(columns={'ID': 'participant', 'Age': 'age'}, inplace=True)
    # df['Participant No'] = df['Participant No'].astype(int)
    return df


def interp(x, x_axis, y_values):
    """
    Performs linear interpolation

    Args:
        x:
        x_axis:
        y_values:

    Returns:
        y

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
