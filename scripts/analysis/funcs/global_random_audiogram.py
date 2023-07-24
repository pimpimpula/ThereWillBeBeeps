import itertools

import pandas as pd
import numpy as np
from typing import Dict, List

from sklearn.linear_model import LogisticRegression

from scripts.API_access import *
from scripts.figure_params import *

from scripts.preprocessing.funcs.resample_audiograms import resample_audiogram
from scripts.analysis.funcs.plots import VisualChecksP50


class DataFormatter:

    def __init__(self):
        pass

    @staticmethod
    def shorten_data(thresholds_data: pd.DataFrame, freqs_3afc):
        """
        Compacts the audiograms dataframe so that it has one row per audiogram, i.e. 10 rows by participant
        (groups by 'participant', 'paradigm', 'pred', and converts 'thresholds' into a list).

        Args:
            thresholds_data (pd.DataFrame): The input dataframe, with one row per frequency in the audiogram.
            freqs_3afc:

        Returns:
            short_data (pd.DataFrame): The output dataframe, shortened such that it has one row per
            audiogram. The 'thresholds' column is converted to a list.
        """

        # Convert the 'thresholds' into list grouped by 'participant', 'paradigm', 'pred'
        df_grouped = thresholds_data.groupby(['participant', 'paradigm', 'pred'])['thresholds'].apply(list).reset_index()

        # Merge the grouped dataframe with the original dataframe to get other columns
        short_data = pd.merge(thresholds_data.drop('thresholds', axis=1), df_grouped,
                              how='inner', on=['participant', 'paradigm', 'pred'])

        short_data.drop_duplicates(subset=['participant', 'paradigm', 'pred'], keep='first', inplace=True)

        # Reorder columns (useless but helps with my undiagnosed OCD)
        cols = short_data.columns.tolist()
        cols.insert(10, cols.pop(cols.index('thresholds')))
        short_data = short_data[cols]

        # Fix frequencies column
        short_data.frequencies = [freqs_3afc] * len(short_data)

        return short_data

    @staticmethod
    def lengthen_data(short_data: pd.DataFrame):
        """
        Explodes the data from short format to long, such that each trial has a row.

        Args:
            short_data (pd.DataFrame): The input dataframe. It should have one row per audiogram,
            and the 'thresholds' column should contain lists.

        Returns:
            all_trials_data (pd.DataFrame): The output dataframe. It has one row per trial.
        """

        all_trials_data = pd.DataFrame()
        for idx, row in short_data.iterrows():
            # number of tones tested for this audiogram (includes repeated init phase)
            ntones = len(row.tested_frequencies)

            # Find length of init phase data
            len_init = short_data.loc[(short_data.paradigm == 'Bayesian')
                                      & (short_data.participant == row.participant), 'len_init'].iloc[0]

            # Get index of each presented tone
            tone_indices = np.linspace(0, ntones - 1, ntones, dtype=int)

            # Format data
            audiogram_data = {'participant': [row.participant] * ntones,
                              'paradigm': [row.paradigm] * ntones,
                              'pred': [row.pred] * ntones,
                              'time_pred': [row.time_pred] * ntones,
                              'freq_pred': [row.freq_pred] * ntones,
                              'tested_frequencies': row.tested_frequencies,
                              'tested_levels': row.tested_levels,
                              'responses': row.responses,
                              'n_tone': tone_indices,
                              'len_init': [None if row.paradigm == '3AFC' else len_init] * ntones,
                              'is_init': [None] * ntones if row.paradigm == '3AFC' else
                              [True if tone_idx < len_init else False for tone_idx in tone_indices],
                              'frequencies': [list(np.geomspace(125, 8000, num=len(row.thresholds)))] * ntones,
                              'thresholds': [row.thresholds] * ntones,
                              'mean_threshold': [row.mean_threshold] * ntones
                              }

            all_trials_data = pd.concat([all_trials_data, pd.DataFrame(audiogram_data)])

        return all_trials_data

    @staticmethod
    def add_distances_to_short_data(short_data: pd.DataFrame, trials_data: pd.DataFrame):
        """
        Adds 'random_threshold', 'random_distance', 'n_tone', and 'is_init' columns to the 'short_data' dataframe.
        Each of these columns contains a list of the corresponding values from the 'trials_data' dataframe
        for each unique combination of paradigm, pred, and participant.

        Args:
            short_data (pd.DataFrame): One row per audiogram.
            trials_data (pd.DataFrame): One row per trial.

        Returns:
            short_data (pd.DataFrame): With added columns 'random_threshold', 'random_distance', 'n_tone', and 'is_init'.
        """

        agg_cols = ['random_threshold', 'random_distance', 'n_tone', 'is_init']

        # Add new data
        new_data = trials_data.groupby(['paradigm', 'pred', 'participant'])[agg_cols].agg(list)
        new_data.reset_index(inplace=True)
        short_data = pd.merge(short_data, new_data, on=['paradigm', 'pred', 'participant'])

        # Reorder the columns (still useless, but it feels good)
        cols = short_data.columns.tolist()
        for col in reversed(agg_cols):  # reverse the order to keep the same order
            cols.insert(8, cols.pop(cols.index(col)))
        short_data = short_data[cols]

        return short_data

    @staticmethod
    def sigmoid_to_df(paradigm: str, pred: str, participant: str,
                      xsigmoid, sigmoid, p50: float,
                      problematic_participants: Dict[str, List[str]]):
        """
        Create a dataframe containing information on 'psychometric' curves.

        Args:
            paradigm (str): Experimental paradigm.
            pred (str): Prediction condition within the current paradigm.
            participant (str): Identifier of the participant.
            xsigmoid (list): Range of tested tones for the current paradigm.
            sigmoid (list): The logistic regression predictions over xsigmoid.
            p50 (float): Computed p50 value.
            problematic_participants (dict):

        Returns:
            pd.DataFrame: A dataframe containing information on 'psychometric' curves for the given participant and condition.
        """

        paradigm_problematic_participants = list(itertools.chain(*problematic_participants.values()))

        this_curve = pd.DataFrame({
            "paradigm": [paradigm] * len(xsigmoid),
            "pred": [pred] * len(xsigmoid),
            "participant": participant,
            "random_distance": xsigmoid,
            "sigmoid_probas": sigmoid,
            "distance_p50": [p50] * len(xsigmoid),
            "problematic": [True if participant in paradigm_problematic_participants else False] * len(xsigmoid)
        })
        return this_curve


class DataProcessor:

    @staticmethod
    def request_global_audiograms(random_data: pd.DataFrame, resampling_frequencies: List):
        """
        Removes the initialization phase trials from the random condition of Continuous and Cluster.
        Makes API requests to get global random audiograms for each participant.
        Resamples the newly obtained audiograms.

        Args:
            random_data (pd.DataFrame):
                df containing the trials data from the 3 random conditions (Randomized, Continuous R, Cluster R)
            resampling_frequencies (list):
                List of frequencies at which the audiograms should be resampled.

        Returns:
            random_audiograms (pd.DataFrame): Contains the resampled audiograms for each participant.
        """

        api_access = get_API_access()

        random_audiograms = pd.DataFrame()
        for participant, participant_data in random_data.groupby('participant'):
            print(f"Computing global random audiogram for {participant}...")

            len_init = participant_data.len_init.unique()[0]

            # remove initialization phase data from Continuous and Cluster points (to avoid tripling points)
            freqs_without_init, levels_without_init, responses_without_init, ntones = [], [], [], []

            for paradigm, paradigm_data in participant_data.groupby('paradigm'):
                # Determine slicing index based on paradigm
                index = 0 if paradigm == 'Bayesian' else len_init

                # Fetch data and append to lists
                freqs = paradigm_data.tested_frequencies.iloc[index:].tolist()
                levels = paradigm_data.tested_levels.iloc[index:].tolist()
                responses = paradigm_data.responses.iloc[index:].tolist()

                # Store tones info
                freqs_without_init.extend(freqs)
                levels_without_init.extend(levels)
                responses_without_init.extend(responses)
                ntones.append([paradigm, len(freqs)])

            # make API request with selected tones info
            x_data = np.array([freqs_without_init, levels_without_init]).T.tolist()
            y_data = [[1] if resp == 1 else [-1] for resp in responses_without_init]
            api_results = make_api_request(x_data, y_data, *api_access)

            # resample audiograms to 100 datapoints
            resampled_results = resample_audiogram(api_results, resampling_frequencies)

            # format data for df
            random_audiogram = pd.DataFrame(
                {'participant': [participant],
                 'random_tested_frequencies': [freqs_without_init],
                 'random_tested_levels': [levels_without_init],
                 'random_responses': [responses_without_init],
                 'random_thresholds': [resampled_results['audiogram']['estimation']],
                 'frequencies': [resampling_frequencies],
                 'random_ntones': [ntones],
                 }
            )

            random_audiograms = pd.concat([random_audiograms, random_audiogram])

        print("All participants processed! Phew :)")

        return random_audiograms

    @staticmethod
    def checks_and_p50_computation(sigmoid: np.ndarray, xsigmoid: np.ndarray,
                                   participant_trials: pd.DataFrame,
                                   problematic_participants: Dict[str, List[str]],
                                   subplot_counters, axs_inverted, axs_p50):

        """
        Checks for anomalies in sigmoid fitting and computes the p50 value.

        Identify if the sigmoid curve is inverted or if the p50 value
        is outside the expected range. In such cases, plot the problematic p50
        estimation and set the p50 value to NaN.

        Parameters
        ----------
        sigmoid : np.ndarray
            The predicted sigmoid curve values.
        xsigmoid : np.ndarray
            The corresponding x values of the sigmoid.
        participant_trials : pd.DataFrame
            Grouped participant trials data for a specific paradigm and pred.
        problematic_participants : list
            List of participants with problematic sigmoid or p50.
        subplot_counters : dict
            Dictionary to keep track of subplot indices for 'Inverted sigmoid' and 'p50 outside range'.
        axs_inverted : list
            List of axes for plotting inverted sigmoid plots.
        axs_p50 : list
            List of axes for plotting p50 outside range plots.

        Returns
        -------
        p50 : float
            The p50 value. It will be NaN if anomalies are detected.
        problematic_participants : list
            Updated list of participants with problematic sigmoid or p50.
        subplot_counters : dict
            Updated dictionary of subplot indices for 'Inverted sigmoid' and 'p50 outside range'.
        """

        paradigm = participant_trials.paradigm.unique()[0]
        pred = participant_trials.pred.unique()[0]
        participant = participant_trials.participant.unique()[0]

        inverted_sigmoid = True if sigmoid[-1] - sigmoid[0] < 0 else False

        p50 = np.interp(.5, sigmoid[::-1] if inverted_sigmoid else sigmoid,
                        xsigmoid[::-1] if inverted_sigmoid else xsigmoid)

        p50_outside_range = False if participant_trials.random_distance.min() <= p50 < participant_trials.random_distance.max() else True

        # Checking for anomalies in the sigmoid fit and p50 range
        if inverted_sigmoid or p50_outside_range:

            anomaly_type = 'Inverted sigmoid' if inverted_sigmoid else 'p50 outside range'
            print(f" - {participant}, {paradigm}{'' if paradigm in ['Bayesian', '3AFC'] else f' ({pred})'}")
            print(f"Anomaly detected: {f'Inverted sigmoid (max - min) = {np.round(sigmoid[-1] - sigmoid[0], 3)}' if sigmoid[-1] - sigmoid[0] < 0 else f'p50 ({np.round(p50, 3)}) outside of tested dB range ({[np.round(participant_trials.random_distance.min(), 3), np.round(participant_trials.random_distance.max(), 3)]})'}")

            # Get next subplot axis from axs using the counter
            ax = axs_inverted[subplot_counters['Inverted sigmoid']] if inverted_sigmoid else axs_p50[subplot_counters['p50 outside range']]
            subplot_counters[anomaly_type] += 1

            # Plot the problematic p50 estimation
            VisualChecksP50.plot_problematic_p50s(ax, sigmoid, xsigmoid, p50, participant_trials, anomaly_type, problematic_participants[anomaly_type])

            # Set p50 to NaN and add participant to list of problematic participants
            p50 = np.nan
            if participant not in problematic_participants[anomaly_type]:
                problematic_participants[anomaly_type].append(participant)
            print(f"p50 set to NaN and {participant} {'added to' if participant not in problematic_participants[anomaly_type] else 'already in'} list of problematic participants.\n")

        return p50, problematic_participants, subplot_counters

    def get_sigmoid_and_p50(self, paradigm_group: pd.DataFrame, paradigm: str, problematic_participants: Dict[str, List[str]],
                            subplot_counters, axs_inverted, axs_p50):
        """
        Analyze data for each participant in the paradigm group.

        Args:
            paradigm_group (pd.DataFrame): DataFrame containing the trial data for a specific paradigm.
            paradigm (str): Experimental paradigm.
            problematic_participants (dict): Dictionary tracking participants with anomalies for this paradigm.
            subplot_counters (dict): Dictionary tracking the current subplot for each type of anomaly.
            axs_inverted, axs_p50 (list): Lists of subplots for anomalies.

        Returns:
            pd.DataFrame: DataFrame containing the 'psychometric' curves data for all participants in the paradigm group.
            dict: Updated dictionary of problematic participants.
            dict: Updated dictionary of subplot counters.
        """

        # Set all the sigmoids' x-axis
        # predict probas for the range of tested tones for the current paradigm (across all participants)
        xsigmoid = np.linspace(paradigm_group.random_distance.min(), paradigm_group.random_distance.max(), num=100)

        pseudo_psychometric_curves = pd.DataFrame()

        for pred, pred_group in paradigm_group.groupby('pred'):
            for participant, participant_group in pred_group.groupby('participant'):
                logreg = LogisticRegression()
                logreg.fit(participant_group.random_distance.values.reshape(-1, 1), participant_group.responses)

                sigmoid = logreg.predict_proba(xsigmoid.reshape(-1, 1))[:, 1]

                p50, problematic_participants, subplot_counter = self.checks_and_p50_computation(
                    sigmoid, xsigmoid, participant_group, problematic_participants,
                    subplot_counters, axs_inverted, axs_p50)

                this_curve = DataFormatter.sigmoid_to_df(paradigm, pred, participant, xsigmoid, sigmoid, p50, problematic_participants)
                pseudo_psychometric_curves = pd.concat([pseudo_psychometric_curves, this_curve])

        return pseudo_psychometric_curves, problematic_participants, subplot_counters
