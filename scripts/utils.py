import os
from pathlib import Path


def get_path(folder_name):
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
    dictionary = {"both": "FT",
                  "frequency": "F",
                  "time": "T",
                  "none": "R"
                  }
    return dictionary[pred]


def exclude_participants():
    """
    Exclude participants:

    - `tvzljm` lied about their age and did not finish the experiment
    - `lkbxgs` had data issues (no Bayesian audiogram saved, all the following experiments were based on the wrong audiogram)
    - `wquuex` had > 20 dB HL
    - `bihhjl` is the HL participant
    - `tyrfqt`, `ikieoz`, `gtyzck` and `ttuwra` did a former version of the 3-AFC task
    """

    return ['tvzljm', 'lkbxgs', 'wquuex', 'bihhjl', 'tyrfqt', 'ikieoz', 'gtyzck', 'ttuwra', '.DS_Store']
