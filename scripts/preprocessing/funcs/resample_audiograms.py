import os
import glob
import pickle
import re
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from scripts.API_access import *
from scripts.figure_params import get_color


def set_audiograms_directory(aud_path, paradigm):
    aud_dir = os.path.join(aud_path, paradigm)
    if not os.path.exists(aud_dir):
        print(f"Created folder {aud_dir}")
        os.makedirs(aud_dir)
    return aud_dir


def fetch_audiogram_data(data_folder, participant, paradigm, pred=None, init=False):
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

    os.chdir(os.path.join(data_folder, participant, paradigm))

    # Find the most recent audiogram matching the tag
    pkl_files = list(filter(os.path.isfile, glob.glob(tag)))
    pkl_files.sort(key=lambda x: os.path.getmtime(x))  # sort by modified time

    if pkl_files:
        print(f"Opening audiogram for {participant}: ({paradigm if pred is None else f'{paradigm}/{pred}'})...")
        print(pkl_files[-1])

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


def recompute(paradigm, participant, participant_data, participant_bayesian_data, url_api, headers,
              remove_initialization=False):
    """
    Recompute audiograms for a specific participant after applying corrections:
    recalibration for 'eqdcwr' and removal of initialization-phase data if requested.

    Args:
    paradigm (str): The paradigm used ('Bayesian', 'Cluster', or 'Continuous').
    participant (str): The identifier for the participant.
    participant_data (dict): The participant's original data.
    participant_bayesian_data (dict): The participant's Bayesian data.
    url_api (str): The API URL.
    headers (dict): The headers to use for the API call.
    remove_init (bool, optional): If True, the initialization data is removed. Defaults to False.

    Returns:
    dict: The recomputed data for the participant.
    """

    # Recalibrate data for 'eqdcwr'
    if participant == 'eqdcwr' and paradigm in ['Bayesian', 'Cluster']:
        # Fix calibration issue with one participant
        tested_points = np.array(participant_data['x'])
        tested_points[:, 1] -= 3.3  # Correct calibration issue
        x_list = [list(point) for point in tested_points]
    else:
        x_list = [list(point) for point in np.array(participant_data['x'])]

    if remove_initialization:
        x_data = x_list[len(participant_bayesian_data['init']['x']) + 1:]
        y_data = participant_data['y'][len(participant_bayesian_data['init']['y']) + 1:]
        recomputed_data = make_api_request(x_data, y_data, url_api, headers)
    else:
        if participant == 'eqdcwr' and paradigm in ['Bayesian', 'Cluster']:
            x_data = x_list
            y_data = participant_data['y']

            recomputed_data = make_api_request(x_data, y_data, url_api, headers)
            recomputed_data['init'] = participant_bayesian_data['init']
            # recomputed_data['x'] = recomputed_data['x'][:-1]
            print(len(recomputed_data['x']), len(recomputed_data['y']))
        else:
            recomputed_data = copy.deepcopy(participant_data)

    return recomputed_data


def fix_truncated_audiogram(participant, recomputed_participant_data, url_api, headers):
    """
    Recompute the audiogram if it doesn't cover the entire frequency range (125-8K Hz). This situation can occur with very negative threshold values. The function shifts the tested levels by 20 dB HL, re-does the request, then lowers the audiogram by 20 dB HL.

    Args:
    participant (str): The identifier for the participant.
    recomputed (dict): The recomputed data for the participant.
    participant_data (dict): The participant's original data.
    participant_bayesian_data (dict): The participant's Bayesian data.
    remove_init (bool, optional): If True, the initialization data is removed. Defaults to False.

    Returns:
    dict: The further recomputed data for the participant.
    """

    print(f"""Recomputed audiogram of participant {participant} is truncated
    (min freq = {round(min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]), 2)},
    max freq = {round(max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]), 2)}). Recomputing...""")

    # Shift levels by 20 dB HL
    tested_points = np.array(recomputed_participant_data['x'])
    tested_points[:, 1] += 20
    x_list = []
    for element in tested_points:
        x_list.append(list(element))

    recomputed_participant_data = make_api_request(participant, x_list, recomputed_participant_data['y'], url_api,
                                                   headers)

    # Lower the audiogram by 20 dB
    for key in recomputed_participant_data['audiogram'].keys():
        for element in recomputed_participant_data['audiogram'][key]:
            element[1] -= 20

    print("Audiogram minimum frequency:",
          min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]))
    print("Audiogram maximum frequency:",
          max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]))
    print("Audiogram:", recomputed_participant_data['audiogram']['estimation'])

    return recomputed_participant_data


def load_or_recompute_and_save_audiograms(raw_data, bayes_data, paradigm,
                                          ignore_participants, aud_dir,
                                          url_api, headers,
                                          pred=None, remove_initialization=False, overwrite=False):
    """
    Recompute audiograms for each participant and save the audiograms if they have not been saved previously.
    Otherwise, load existing data.

    -------------

    Args:

    - raw_data (dict): Previously saved data for all participants.
    - bayes_data (dict): Previously saved data for the Bayesian paradigm and for all participants.
    - paradigm (str): The paradigm used ('Bayesian', 'Cluster', or 'Continuous').
    - ignore_participants (list): A list of participant identifiers to ignore.
    - aud_dir (str): The directory where the audiograms are saved.
    - pred (str, optional): The predictability condition. Can be 'none', 'time', 'frequency', or 'both'. Defaults to None.
    - remove_initialization (bool, optional): If True, the initialization data is removed. Defaults to False.
    - overwrite (bool, optional): If True, force save audiogram even if the output file already exists.

    Returns:

    - dict: The recomputed data for all participants.

    """

    recomputed = {}
    for participant in raw_data.keys():
        if participant in ignore_participants:
            print(f"Participant {participant} ignored")
        else:

            # Create filename
            audfile_name = f"{participant}_{paradigm}.pkl" if paradigm == 'Bayesian' \
                else f'{participant}_{paradigm}_{pred}.pkl'

            # Save only if audiogram doesn't already exist
            save_audiogram = True
            for filename in os.listdir(aud_dir):
                if re.search(audfile_name, filename):
                    save_audiogram = False
                    audfile_name = filename

            if overwrite:
                save_audiogram = True
            # save_audiogram = True  # Uncomment to force rewriting
            # save_audiogram = True  if participant in ['hnxrev'] else False
            # save_audiogram = True if paradigm == 'Continuous' else False

            if save_audiogram:

                # apply calibration correction if necessary
                recomputed[participant] = recompute(paradigm, participant, raw_data[participant],
                                                    bayes_data[participant],
                                                    url_api, headers,
                                                    remove_initialization=remove_initialization)

                # recompute audiogram if truncated
                if min(np.array(recomputed[participant]['audiogram']['estimation'])[:, 0]) > 126:
                    recomputed[participant] = fix_truncated_audiogram(participant, recomputed[participant],
                                                                      url_api, headers)

                # save it
                if paradigm == 'Bayesian':
                    print(f"Saving audiogram of {participant} in {os.path.join(*aud_dir.split('/')[-5:])}...")
                else:
                    print(f"Saving {pred} audiogram of {participant} in {os.path.join(*aud_dir.split('/')[-5:])}...")

                audfile = open(os.path.join(aud_dir, audfile_name), "wb")
                pickle.dump(recomputed[participant], audfile)  # ['audiogram']
                audfile.close()

            else:
                # fetch previously recomputed data
                print(
                    f"Loading already existing recomputed audiogram of {participant} from {os.path.join(*aud_dir.split('/')[-5:])}")
                audfile = open(os.path.join(aud_dir, audfile_name), "rb")
                recomputed[participant] = pickle.load(audfile)
                audfile.close()

    return recomputed


def plot_recomputed_audiograms(recomputed_audiograms, paradigm, colormap=None, pred=None):
    if pred == 'both':
        color = '#EE2E31'  # '#FF6B6B'
    elif pred == 'time':
        color = '#4ECDC4'
    elif pred == 'frequency':
        color = '#EFC352'  # '#FCD7AD'
    elif pred == 'none':
        color = '#292F36'

    for participant in recomputed_audiograms.keys():
        # plt.subplot(int(len(list(recomputed.keys()))/2), 2, len(list(recomputed.keys())))

        if pred is None:
            color = colormap.colors[list(recomputed_audiograms.keys()).index(participant)]

        plt.plot(*np.array(recomputed_audiograms[participant]['audiogram']['estimation']).T,
                 color=color, alpha=1, label=f"{participant}: {pred}")
        plt.title(f"{paradigm}" if paradigm == 'Bayesian' else f"{paradigm} - {pred}")

        plt.xscale('log')
        # plt.legend()
        plt.xticks([125, 1000, 8000], [125, 1000, 8000])

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Threshold (dB HL)")


def resample_audiogram(audiogram_dict, xfreqs):
    """
    Resample the given audiogram to n = len(xfreqs) data points (for us, n = 100).

    Args:
        audiogram_dict (dict): A dictionary containing the audiogram, such as the API request results. audiogram_dict['audiogram'] should contain fields 'estimation', 'plus_line' and 'minus_line'
        xfreqs (list): The frequencies at which to sample the audiogram.

    Returns:
        dict: The resampled audiogram.
    """

    # resampled_audiogram = {}
    resampled_audiogram = copy.deepcopy(audiogram_dict)

    # RESAMPLE AUDIOGRAM TO 100 DATAPOINTS
    for line in ['estimation', 'plus_line', 'minus_line']:
        interp = np.array(audiogram_dict['audiogram'][line])
        linear_interp = interp1d(interp[:, 0], interp[:, 1],
                                 fill_value="extrapolate"  # just to fix lower bound issue (some values lie between 125 and 126)
                                 )
        resampled_audiogram['audiogram'][line] = linear_interp(xfreqs)

    return resampled_audiogram


def load_or_resample_and_save_audiograms(paradigm, recomputed_data, xfreqs, remove_initialization, aud_dir, pred=None,
                                         overwrite=False):
    """
    Resample audiograms for each participant and save the audiograms if they have not been saved previously.
    Otherwise, load existing data.

    Args:

    - paradigm (string): The current paradigm.
    - recomputed_data (dict): Previously recomputed data for all participants.
    - xfreqs (numpy array): Array of frequencies for resampling.
    - remove_initialization (bool): If True, the initialization data is removed.
    - aud_dir (string): Directory where the audiograms are saved for this paradigm.
    - overwrite (bool, optional): If True, force save audiogram even if the output file already exists.

    Returns:

    - dict: The resampled data for all participants.

    """

    print("----------------", paradigm, "----------------")
    resampled_data = {}
    for participant in list(recomputed_data.keys()):
        # Create filename
        audfile_name = f"{participant}_{paradigm if pred is None else f'{paradigm}_{pred}'}_resampled.pkl"

        # Check if audiogram already exists
        save_audiogram = True
        for filename in os.listdir(aud_dir):
            if re.search(audfile_name, filename):
                save_audiogram = False
                audfile_name = filename

        if overwrite:
            save_audiogram = True

        if save_audiogram:
            print(f"Saving resampled audiogram of {participant} in {os.path.join(*aud_dir.split('/')[-5:])}...")

            # Resample audiogram
            resampled_data[participant] = resample_audiogram(recomputed_data[participant], xfreqs)

            # Save it
            audfile = open(os.path.join(aud_dir, audfile_name), "wb")
            pickle.dump(resampled_data[participant], audfile)
            audfile.close()
        else:
            # Load previously resampled data
            print(
                f"Loading already existing resampled audiogram of {participant} from {os.path.join(*aud_dir.split('/')[-5:])}")
            audfile = open(os.path.join(aud_dir, audfile_name), "rb")
            resampled_data[participant] = pickle.load(audfile)
            audfile.close()

    return resampled_data


def plot_resampled_audiograms(recomputed_audiograms, resampled_audiograms, xfreqs, paradigm, pred=None):
    """
    Plot the resampled audiograms.

    Args:
        audiogram_dict (dict): A dictionary containing the original audiograms.
        resampled_audiograms (dict): A dictionary containing the resampled audiograms.
        xfreqs (list): The frequencies at which the audiograms were sampled.
        AFC_freqs (list): The frequencies at which the AFC was measured.
        paradigm (str): The paradigm used.
        pred (str, optional): The prediction model used. Can be 'none', 'time', 'frequency', or 'both'. Defaults to None.

    Returns:
        None
    """

    color = get_color(paradigm, pred)

    for participant in list(resampled_audiograms.keys()):  # in ["rumszd"]:  # in list(bayes_data.keys()):

        plt.plot(*np.array(recomputed_audiograms[participant]['audiogram']['estimation']).T,
                 color='black' if pred is None else color,
                 alpha=0.2,
                 )
        plt.scatter(xfreqs, resampled_audiograms[participant]['audiogram']['estimation'],
                    color='black' if pred is None else color,
                    alpha=0.2,
                    s=15 if paradigm == 'Bayesian' else 5)

    plt.xscale('log')
    plt.xticks([125, 1000, 8000], [125, 1000, 8000])
    if paradigm == 'Bayesian':
        plt.title(f"{paradigm} audiograms")
    else:
        plt.title(f"{paradigm} - {pred} audiograms")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Threshold (dB HL)")
    # plt.show()
