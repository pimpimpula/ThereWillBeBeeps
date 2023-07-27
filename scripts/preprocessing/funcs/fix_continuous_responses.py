import copy
import pickle

from scripts.API_access import *
from scripts.utils import *


def fetch_initialization_data(participant, dataFolder):
    """
    Fetch the initialization phase data for a participant.

    Args:
        participant (str): Participant's identifier.
        dataFolder (str): Directory where the participant's data is stored.

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
    os.chdir(dataFolder)
    return init_data


def check_false_alarm_rates(participant_data, shifted_responses=False):
    """
    Calculate false alarm rate for a participant.

    Args:
        participant_data (DataFrame): The trial data for this participant.
        shifted_responses (bool, optional): Whether to use shifted responses. Defaults to False.

    Returns:
        float: False alarm rate for the participant.
    """
    CT_data = participant_data.loc[participant_data.isCatchTrial == 1]
    FA_count = len(CT_data.loc[CT_data['shifted_answers' if shifted_responses else 'feedback.keys'] != 'None'])
    FAR = FA_count / len(CT_data)
    print(f"{'Shifted' if shifted_responses else 'Original'} false alarm rate: {np.round(FAR, 3)}")
    return FAR


def correct_data(participant_data, participant):
    """
    Correct data issues for a participant. This includes fixing negative reaction times,
    removing catch trials, and removing the first two trials of each sweep.
    Keeps track of whether the corrections improve FAR and detection ratios.

    Args:
        participant_data (DataFrame): The participant's data.
        participant (str): Participant's identifier.

    Returns:
        DataFrame: The corrected data for the participant.
        list: Changes in false alarm rate due to corrections.
        list: Changes in detection ratio due to corrections.
    """

    better_FAR = []
    better_DR = []

    participant_data = participant_data.rename(columns={'feedback.rt': 'reaction_times'})  # inplace=True)
    original_detection_ratio = np.round((len(participant_data[participant_data['feedback.keys'] != 'None']) / len(participant_data)), 3)

    # Fix dtype of RTs
    participant_data.reaction_times = [[float(rt.strip()) for rt in str(response).strip('[]').split(',')] if response else np.NaN for response in participant_data.reaction_times]
    participant_data.insert(len(participant_data.columns), 'shifted_answers', [None] * len(participant_data))

    # Check for negative RTs
    RTs = np.array([rt for rt_values in participant_data.reaction_times.values for rt in rt_values])

    if any(RTs < 0):  # Fix negative reaction times
        print(f"Fixing negative reaction times for {participant}: mean RT = {np.nanmean(RTs).round(3)}, {np.sum(RTs < 0)} RTs < 0")

        # Check false alarm rate previous to shifting
        old_FAR = check_false_alarm_rates(participant_data, shifted_responses=False)

        # Shift all answers to the previous trial
        participant_data['shifted_answers'] = participant_data['feedback.keys'].shift(-1)

        # Remove last trial
        participant_data = participant_data.drop(participant_data.index[-1])

        # Check false alarm rate after shifting
        new_FAR = check_false_alarm_rates(participant_data, shifted_responses=True)
        better_FAR.append(old_FAR - new_FAR)

    else:
        print(f"no need to fix {participant}? mean RT = {np.nanmean(RTs).round(3)}, RTs < 0 = {np.sum(RTs < 0)}")

        participant_data['shifted_answers'] = participant_data['feedback.keys']

        # Check false alarm rate
        check_false_alarm_rates(participant_data)

    # Remove catch trial data
    participant_data = participant_data.loc[participant_data.isCatchTrial == 0]

    # Remove the first two trials of predictive sweeps
    og_len = len(participant_data)
    participant_data = participant_data.loc[~((participant_data['trials.thisN'] < 2) & (participant_data['Prediction'] != 'none'))]
    print('-----')
    print(f"Removing first 2 trials of each sweep (=/= none): {og_len} -> {len(participant_data)} total trials")

    # Check Detection Ratios
    new_detection_ratio = np.round((len(participant_data[participant_data.shifted_answers != 'None']) / len(participant_data)), 3)
    better_DR.append(new_detection_ratio - original_detection_ratio)

    print('-----')
    print(f"Original detection ratio: {original_detection_ratio}")
    print(f"New detection ratio: {new_detection_ratio}")
    print('-----')

    return participant_data, better_FAR, better_DR


def recompute_audiogram(participant, pred, pred_group, init_data, API_calls, url_api, headers):
    """
    Recompute the audiogram for a participant using the API.

    Args:
        participant (str): Participant's identifier.
        pred (str): Condition of prediction.
        pred_group (DataFrame): The participant's data for the given condition.
        init_data (dict): The participant's initialization data.
        API_calls (dict): Existing API calls for the participant.
        url_api (str): The API URL.
        headers (dict): The headers to use for the API call.

    Returns:
        dict: The API response.
    """

    # Initialize audiogram
    API_calls[participant][pred] = copy.deepcopy(init_data)

    # Format the corrected data (minus the first two tones in a sweep) to the one requested by the API
    for idx, row in pred_group.iterrows():
        API_calls[participant][pred]['x'].append([row['Frequency'], row['Level']])
        API_calls[participant][pred]['y'].append([1] if row['shifted_answers'] != "None" else [-1])
    API_calls[participant][pred]['is_init_phase'] = False

    # print(len(API_calls[participant][pred]['y']))
    print(f"Making request for {participant} ({pred})")
    # req = requests.post(url=url_api, headers=headers, json=API_calls[participant][pred])
    API_answer = make_api_request(API_calls[participant][pred]['x'],
                                  API_calls[participant][pred]['y'],
                                  url_api, headers)

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


'''
def process_participants(data_folder, ignore_participants, overwrite=False):
    """
    Preprocess the Continuous raw data for all participants.

    For each participant, the function fetches their initialization data, processes their CSV files from the Continuous
    task, corrects the data, recomputes the audiogram using the API, and saves the recomputed audiogram. The function
    can optionally overwrite existing audiograms.

    Args:
        data_folder (str): Directory where the participant's data is stored.
        ignore_participants (list of str): List of participant identifiers to ignore.
        overwrite (bool, optional): Whether to overwrite existing audiograms. Defaults to False.

    Returns:
        None
    """

    url_api, headers = get_API_access()

    os.chdir(data_folder)

    API_calls = {}
    API_answer = {}
    init_data = {}

    # Keep track of whether detection and false alarm ratios improve
    are_FARs_looking_better = []
    are_DRs_looking_better = []

    # Get the list of participants
    participants = exclude_participants(os.listdir(get_path('raw_data')))

    for participant in participants:
        API_calls[participant] = {}
        API_answer[participant] = {}

        recomputed_files = filter_csv_files(participant, 'Continuous', "*_fixed.pkl")

        if len(recomputed_files) == 4 and not overwrite:  #TODO: improve this?
            pass
        else:
            print("\n--------------------", participant, "--------------------")

            init_data[participant] = fetch_initialization_data(participant, data_folder)

            participant_data = process_csv_files(participant)

            participant_data, effect_on_FAR, effect_on_DR = correct_data(participant_data, participant)

            are_FARs_looking_better.append(effect_on_FAR)
            are_DRs_looking_better.append(effect_on_DR)

            for pred, pred_group in participant_data.groupby('Prediction'):

                API_answer[participant][pred] = recompute_audiogram(participant, pred, pred_group,
                                                                    init_data[participant], API_calls,
                                                                    url_api, headers)

                save_audiogram(data_folder, participant, pred, API_answer[participant][pred])

            print("-------------------------------------------------")

    # See if we've improved FARs and detection rations with the corrections made
    print("")
    print(f"Improved (decreased) FARs for {np.sum(np.array(are_FARs_looking_better) < 0)} participants")
    print(f"Mean improvement: {np.round(np.mean(are_FARs_looking_better) * 100)} %")
    print("")
    print(f"Improved (increased) detection ratios for {np.sum(np.array(are_DRs_looking_better) > 0)} participants")
    print(f"Mean improvement: {np.round(np.mean(are_DRs_looking_better) * 100)} %")
    print("")'''