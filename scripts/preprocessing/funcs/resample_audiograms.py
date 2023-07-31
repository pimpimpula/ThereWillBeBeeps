import re
import copy

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from scripts.API_access import *
from scripts.utils import *

from scripts.figure_params import get_color, pred_palette


class PreprocMain:

    def __init__(self):
        self.API_access = get_API_access()

    def recompute_audiogram(self, paradigm, participant, participant_data,
                            participant_bayesian_data, remove_initialization=False):
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
            print("Recomputing audiogram after removing trials from the initialization phase...")
            x_data = x_list[len(participant_bayesian_data['init']['x']) + 1:]
            y_data = participant_data['y'][len(participant_bayesian_data['init']['y']) + 1:]
            recomputed_data = make_api_request(x_data, y_data, *self.API_access)

        else:
            if participant == 'eqdcwr' and paradigm in ['Bayesian', 'Cluster']:

                print("Apply calibration correction to eqdcwr data...")

                x_data = x_list
                y_data = participant_data['y']

                recomputed_data = make_api_request(x_data, y_data, *self.API_access)
                recomputed_data['init'] = participant_bayesian_data['init']
                # recomputed_data['x'] = recomputed_data['x'][:-1]
                print(len(recomputed_data['x']), len(recomputed_data['y']))

            else:
                recomputed_data = copy.deepcopy(participant_data)

        return recomputed_data

    def fix_truncated_audiogram(self, participant, recomputed_participant_data):
        """
        Recompute the audiogram if it doesn't cover the entire frequency range (125-8K Hz).
        This situation can occur with very negative threshold values, and is only a problem when recomputing the
        audiograms without initialization data.
        The function shifts the tested levels by 30 dB HL, re-does the request,
        then lowers back the audiogram by 30 dB HL.

        Args:
            participant (str): The identifier for the participant.
            recomputed_participant_data (dict): The recomputed data for the participant.

        Returns:
        dict: The further recomputed data for the participant.
        """

        print("-----")
        min_freq = min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0])
        max_freq= max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0])
        print(f"Recomputed audiogram of participant {participant} is truncated"
        f"(min freq = {round(min_freq, 2)} Hz, max freq = {round(max_freq, 2)} Hz)")

        min_lvl = min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 1])
        max_lvl = max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 1])
        print(f"(min level = {round(min_lvl, 2)} dB HL, max level = {round(max_lvl, 2)} dB HL)")
        print("Try to shift levels by 30 dB and recompute...")

        # Raise levels by 30 dB HL
        tested_points = np.array(recomputed_participant_data['x'])
        tested_points[:, 1] += 30 if max_lvl < 120 else -30
        x_list = [list(tone_info) for tone_info in tested_points]

        print("Making API request...")
        recomputed_participant_data = make_api_request(x_list, recomputed_participant_data['y'],
                                                       *self.API_access)

        print("Shift audiogram levels back by 30 dB...")
        # Lower the audiogram by 30 dB
        recomputed_participant_data['audiogram'] = {
            key: [[element[0], element[1] - 30] for element in value]
            for key, value in recomputed_participant_data['audiogram'].items()
        }

        print(f"""After recomputing:
        (min freq = {round(min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]), 2)} Hz,
        max freq = {round(max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 0]), 2)} Hz)""")
        print(f"(min level = {round(min(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 1]), 2)} dB,"
              f"max level = {round(max(np.array(recomputed_participant_data['audiogram']['estimation'])[:, 1]), 2)} dB)")
        print('-----')

        return recomputed_participant_data

    def load_or_recompute_and_save_audiograms(self, raw_data, bayes_data, paradigm,  aud_dir,
                                              pred=None, remove_initialization=False, overwrite=False):
        """
        Recompute audiograms for each participant and save the audiograms if they have not been saved previously.
        Otherwise, load existing data.

        -------------

        Args:

        - raw_data (dict): Previously saved data for all participants.
        - bayes_data (dict): Previously saved data for the Bayesian paradigm and for all participants.
        - paradigm (str): The paradigm used ('Bayesian', 'Cluster', or 'Continuous').
        - aud_dir (str): The directory where the audiograms are saved.
        - pred (str, optional): The predictability condition. Defaults to None.
        - remove_initialization (bool, optional): If True, the initialization data is removed. Defaults to False.
        - overwrite (bool, optional): If True, force save audiogram even if the output file already exists.

        Returns:

        - dict: The recomputed data for all participants.

        """

        recomputed = {}
        for participant in raw_data.keys():

            print("----------------------------------------------------")
            print(participant)

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

            if save_audiogram:

                # apply calibration correction if necessary
                recomputed[participant] = self.recompute_audiogram(paradigm, participant, raw_data[participant],
                                                                   bayes_data[participant], remove_initialization)

                # recompute audiogram if truncated
                if min(np.array(recomputed[participant]['audiogram']['estimation'])[:, 0]) > 126 or \
                        max(np.array(recomputed[participant]['audiogram']['estimation'])[:, 0]) < 7999:
                    recomputed[participant] = self.fix_truncated_audiogram(participant, recomputed[participant])

                # save it
                if paradigm == 'Bayesian':
                    print(f"Saving audiogram of {participant} in {os.path.join(*aud_dir.split('/')[-5:])}...")
                else:
                    print(f"Saving {pred} audiogram of {participant} in {os.path.join(*aud_dir.split('/')[-5:])}...")

                audfile = open(os.path.join(aud_dir, audfile_name), "wb")
                pickle.dump(recomputed[participant], audfile)
                audfile.close()

            else:
                # fetch previously recomputed data
                print(f"Loading already existing recomputed audiogram of {participant} "
                      f"from {os.path.join(*aud_dir.split('/')[-5:])}")

                audfile = open(os.path.join(aud_dir, audfile_name), "rb")
                recomputed[participant] = pickle.load(audfile)
                audfile.close()

        return recomputed

    @staticmethod
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
                                     fill_value="extrapolate"
                                     # extrapolate just to fix lower bound issue (some values lie between 125 and 126)
                                     )
            resampled_audiogram['audiogram'][line] = linear_interp(xfreqs)

        return resampled_audiogram

    def load_or_resample_and_save_audiograms(self, paradigm, recomputed_data, xfreqs,
                                             aud_dir, pred=None, overwrite=False):
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

            print("----------------")
            print(participant, '\n')

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
                resampled_data[participant] = self.resample_audiogram(recomputed_data[participant], xfreqs)

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

    @staticmethod
    def plot_recomputed_audiograms(recomputed_audiograms, paradigm, colormap=None, pred=None):

        for participant in recomputed_audiograms.keys():

            if pred is None:
                color = colormap.colors[list(recomputed_audiograms.keys()).index(participant)]
            else:
                color = pred_palette(pred)

            plt.plot(*np.array(recomputed_audiograms[participant]['audiogram']['estimation']).T,
                     color=color, alpha=1, label=f"{participant}: {pred}")
            plt.title(f"{paradigm}" if paradigm == 'Bayesian' else
                      f"{paradigm} - {translate_conditions(pred)}")

            plt.xscale('log')
            plt.xticks([125, 1000, 8000], [125, 1000, 8000])

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Threshold (dB HL)")

    @staticmethod
    def plot_resampled_audiograms(recomputed_audiograms, resampled_audiograms, xfreqs, paradigm, pred=None):
        """
        Plot the resampled audiograms.

        Args:
            recomputed_audiograms (dict): A dictionary containing the original audiograms.
            resampled_audiograms (dict): A dictionary containing the resampled audiograms.
            xfreqs (list): The frequencies at which the audiograms were resampled.
            paradigm (str):
            pred (str, optional): Defaults to None.
        """

        color = get_color(paradigm, pred)

        for participant in list(resampled_audiograms.keys()):

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
        plt.title(f"{paradigm}{'' if paradigm else '- ' + translate_conditions(pred)} audiograms")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Threshold (dB HL)")
