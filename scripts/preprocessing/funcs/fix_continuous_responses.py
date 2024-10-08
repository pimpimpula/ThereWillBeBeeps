import ast
import copy
import re

from scripts.API_access import *
from scripts.utils import *


def check_false_alarm_rates(participant_data, shifted_responses=False):
    """
    Calculate false alarm rate for a participant.

    Args:
        participant_data (DataFrame): The trial data for this participant.
        shifted_responses (bool, optional): Whether to use shifted responses. Defaults to False.

    Returns:
        float: False alarm rate for the participant.
    """

    # look for catch trial data
    ct_data = participant_data.loc[participant_data.isCatchTrial == 1]

    # count the number of false alarms based on the original or corrected responses
    if shifted_responses:
        fa_count = ct_data['corrected_responses'].sum()
    else:
        fa_count = len(ct_data.loc[ct_data['responses'] != 'None'])

    # compute false alarm rate
    far = fa_count / len(ct_data)

    print(f"{'Shifted' if shifted_responses else 'Original'} false alarm rate: {np.round(far, 3)}")
    return far


def check_detection_ratios(are_drs_looking_better, participant_data):
    """
    Could use the same logic as for FARs but that would be way too simple
    """

    # check original detection ratio
    og_detections = len(participant_data.loc[(participant_data.isCatchTrial == 0) &
                                             (participant_data.responses != 'None'), 'responses'])
    original_detection_ratio = og_detections / len(participant_data)

    # check new detection ratio
    new_detections = participant_data.loc[participant_data.isCatchTrial == 0, 'corrected_responses'].sum()
    new_detection_ratio = new_detections / len(participant_data)

    print('-----')
    print(f"Original detection ratio: {np.round(original_detection_ratio, 3)}")
    print(f"New detection ratio: {np.round(new_detection_ratio, 3)}")
    print('-----')

    are_drs_looking_better.append(new_detection_ratio - original_detection_ratio)

    return are_drs_looking_better


def correct_data(participant_data, participant, filter_rts=False):
    """
    Correct data issues for a participant. This includes fixing negative reaction times,
    removing catch trials, and removing the first two trials of each sweep.
    Keeps track of whether the corrections improve FAR and detection ratios.

    Args:
        participant_data (DataFrame): Participant Continuous data.
        participant (str): Participant ID.
        filter_rts (bool): Decide whether to filter RTs < 1 sec. Defaults to False.

    Returns:
        DataFrame: The corrected data for the participant.
        list: Changes in false alarm rate due to corrections.
        list: Changes in detection ratio due to corrections.
    """

    # Fix dtype of RTs
    participant_data = fix_feedbackRT_format(participant_data)

    # Check for negative RTs
    RTs = np.array([rt for rt_values in participant_data['feedback.rt'].values for rt in rt_values])

    print(f"Fixing negative RTs for {participant}: "
          f"mean RT = {np.nanmean(RTs).round(2)}, {np.sum(RTs < 0)}/{len(RTs)} RTs < 0")

    # Correct answers
    corrected_responses = correct_continuous_responses(participant_data, filter_rts)

    # Merge dataframes
    participant_data = pd.merge(participant_data,
                                corrected_responses[['feedback.started', 'corrected_rts', 'corrected_responses']],
                                on='feedback.started')

    # Remove last trial
    participant_data = participant_data.drop(participant_data.index[-1])

    # Remove catch trial data   MOVED OUTSIDE OF FUNCTION TO COMPUTE OVER ALL CSVs
    # participant_data = participant_data.loc[participant_data.isCatchTrial == 0]

    # Remove the first two trials of predictive sweeps
    og_len = len(participant_data)
    participant_data = participant_data.loc[
        ~((participant_data['trials.thisN'] < 2) & (participant_data['pred'] != 'none'))]

    print('-----')
    print(f"Removing first 2 trials of each sweep (=/= none): {og_len} -> {len(participant_data)} total trials")

    return participant_data


def recompute_audiogram(participant, pred, pred_group, init_data, url_api, headers):
    """
    Recompute the audiogram for a participant using the API.

    Args:
        participant (str): Participant ID.
        pred (str): Condition of predictability.
        pred_group (DataFrame): The participant's data for the given condition.
        init_data (dict): The participant's initialization data from the Randomized paradigm.
        url_api (str): The API URL.
        headers (dict): The headers to use for the API call.

    Returns:
        dict: The API response.
    """

    # Initialize audiogram with init data
    API_call = copy.deepcopy(init_data)

    # Format the corrected data for the API
    for idx, row in pred_group.iterrows():
        API_call['x'].append([row['Frequency'], row['Level']])
        API_call['y'].append([1] if row['corrected_responses'] == 1 else [-1])
    API_call['is_init_phase'] = False

    # print(len(API_calls[participant][pred]['y']))
    print(f"Making request for {participant} ({pred})")
    # req = requests.post(url=url_api, headers=headers, json=API_calls[participant][pred])
    API_answer = make_api_request(API_call['x'], API_call['y'], url_api, headers)

    return API_answer


def save_audiogram(data_folder, participant, pred, API_answer):
    """
    Save the recomputed audiogram for a participant.

    Args:
        data_folder (str): Directory where the participant's data is stored.
        participant (str): Participant's identifier.
        pred (str): Type of prediction.
        API_answer (dict): The API response to save.
    """

    filename = f"{participant}_Continuous_{pred}_fixed.pkl"
    output_path = os.path.join(data_folder, participant, 'Continuous', filename)

    print(f"Saving recomputed audiogram: {output_path}")
    f = open(output_path, "wb")
    pickle.dump(API_answer, f)
    f.close()


def parse_log(df_log):
    """
    Parse log to find trials with recorded key presses.
    """

    # THIS IS HORRIBLE AND I'M SORRY BUT IT WORKS?

    # group by 'Created sequence:'
    sequence_groups = df_log.groupby(df_log.message.str.contains('Created sequence:').cumsum())

    # initialize an empty list to hold the data for the new DataFrame
    data = []

    # loop through each sequence group
    for sequence, sequence_group in sequence_groups:
        # keep lines containing 'New trial' or 'Sound' or 'Keypress'
        sequence_group = sequence_group.loc[sequence_group.message.str.contains('New trial|Sound|Keypress')]

        # find the index of the first 'New trial' line
        new_trial_indices = sequence_group[sequence_group.message.str.contains('New trial')].index

        # if no 'New trial' line is found, skip to the next sequence
        if new_trial_indices.empty:
            continue

        # slice the DataFrame from the first 'New trial' line
        sequence_group = sequence_group.loc[new_trial_indices[0]:]

        # find 'New trial' groups within each sequence group
        trial_groups = sequence_group.groupby(sequence_group.message.str.contains('New trial').cumsum())

        # loop through each trial group
        for trial_name, trial_group in trial_groups:

            # get the repetition number from the 'New trial' message
            rep_match = re.search(r'rep=(\d+)', trial_group['message'].values[0])
            rep_number = int(rep_match.group(1))

            # if 'Sound probe started' is not in the group's messages, skip to the next iteration
            if 'Sound probe started' not in trial_group['message'].values:
                continue

            # if the repetition number is less than 2, skip to the next iteration
            if rep_number < 2:
                continue

            # get the timestamp from the first row of the trial group
            trial_timestamp = float(
                trial_group.loc[trial_group.message.str.contains('Sound probe started'), 'timestamp'].values[0])

            # get the reaction times
            trial_rts = [float(rt) - trial_timestamp for rt in
                         trial_group.loc[trial_group['message'].str.contains('Keypress'), 'timestamp']]

            # check if a row exists in trial_group containing 'Keypress: space' and assign the result to 'responses'
            responses = 1 if 'Keypress: space' in trial_group['message'].values else 0

            # create a dictionary with the data for the current trial and append it to the list
            data.append({'log_timestamp': trial_timestamp, 'rep_number': rep_number,
                         'log_responses': responses, 'log_rt': trial_rts})

    # create a new DataFrame from the list of dictionaries
    df_trials = pd.DataFrame(data)
    df_trials.reset_index(inplace=True)
    return df_trials


def fix_reaction_times(row, df):
    """
    Get "real" reaction times from adding the next row's negative reaction time to the start of feedback collection.
    """

    if row.name < len(df) - 1:
        if len(df.loc[row.name + 1, 'feedback.rt']) > 0:
            return [df.loc[row.name + 1, 'feedback.started'] + rt for rt in df.loc[row.name + 1, 'feedback.rt']]
        else:
            return []
    else:
        return []


def filter_rts_under_1sec(row, length):
    """
    Check if at least one of the corrected reaction times for this trial is < 1sec.
    """

    if len(row['corrected_rts']) > 0:  # and row.name < length - 1:
        return 1 if np.array(
            [row['feedback.started'] < rt < row['feedback.started'] + 1 for rt in row['corrected_rts']]).any() else 0
    else:
        return np.nan


def fix_feedbackRT_format(df):
    df['feedback.rt'] = df['feedback.rt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    return df


def correct_continuous_responses(csv_data, filter_rts=False):
    """
    Shifts all Continuous responses from the csv file by one trial. Fixes negative reaction times.
    Filters out detections with corrected RTs > 1 sec after the tone onset if requested.

    *Important*: **Remember to discard the first two tones of non-R sweeps next!**

    Args:
        csv_data (pd.DataFrame):
        filter_rts (bool): Decide whether to filter RTs < 1sec. Defaults to False

    Returns:
        df with the corrected responses in column 'corrected_responses'

    """

    # select the required columns and rename them
    df_trials_csv = csv_data[['feedback.started', 'trials.thisN', 'responses', 'feedback.rt', 'pred']].rename(
        columns={'trials.thisN': 'rep_number'})

    # Fix format issues
    df_trials_csv['feedback.started'] = df_trials_csv['feedback.started'].astype(float)

    # Fix the reaction times
    df_trials_csv['corrected_rts'] = df_trials_csv.apply(lambda x: fix_reaction_times(x, df_trials_csv), axis=1)
    # Count positive detections
    og_detections = df_trials_csv['corrected_rts'].apply(lambda x: len(x) > 0).sum()

    # Filter responses < 1sec after tone onset
    if filter_rts:
        df_trials_csv['corrected_responses'] = df_trials_csv.apply(
            lambda x: filter_rts_under_1sec(x, len(df_trials_csv)), axis=1)

        print(
            f"Filtering RTs < 1sec after the tone onset: "
            f"{int(og_detections - df_trials_csv['corrected_responses'].sum())} detections removed")

    else:
        df_trials_csv['corrected_responses'] = df_trials_csv['corrected_rts'].apply(lambda x: 1 if len(x) > 0 else 0)
        print("NOT filtering RTs < 1sec after the tone onset: NO detections removed")

    return df_trials_csv
