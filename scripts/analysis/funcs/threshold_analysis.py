from scripts.stats import StatsParams
from scripts.utils import *


def audiograms_to_df(paradigm, audiograms_path, xfreqs, pred="Bayesian", no_init=False):
    # Filter PKL preprocessed audiogram files matching the paradigm and pred condition selected
    file_tag = f"*{'' if pred == 'Bayesian' else f'_{pred}'}_resampled.pkl"
    file_list = glob.glob(os.path.join(audiograms_path, paradigm, file_tag))

    paradigm_df = pd.DataFrame()
    paradigm_dict = {}

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        participant = file_name.split('_')[0]
        paradigm_dict[participant] = pd.read_pickle(file_path)
        nrows = len(paradigm_dict[participant]['audiogram']['estimation'])

        participant_df = pd.DataFrame(
            {'participant': [participant] * nrows,
             'paradigm': [paradigm] * nrows,
             'pred': [pred] * nrows,
             'time_pred': [None if paradigm == "Bayesian" else (
                 False if pred in ['none', 'frequency'] else True)] * nrows,
             'freq_pred': [None if paradigm == "Bayesian" else (False if pred in ['none', 'time'] else True)] * nrows,
             'tested_frequencies': [[tested_point[0] for tested_point in paradigm_dict[participant]['x']] for _ in
                                    range(nrows)],
             'tested_levels': [[tested_point[1] for tested_point in paradigm_dict[participant]['x']] for _ in
                               range(nrows)],
             'responses': [[0 if resp == [-1] else 1 for resp in
                            (paradigm_dict[participant]['y'][:-1] if (
                                        paradigm == 'Bayesian' and participant != 'eqdcwr')
                             else paradigm_dict[participant]['y'])]
                           for _ in range(nrows)],
             'len_init': [len(paradigm_dict[participant]['init']['x']) if pred == "Bayesian" and not no_init else None] * nrows,
             'thresholds': paradigm_dict[participant]['audiogram']['estimation'],
             'frequencies': xfreqs,
             'mean_threshold': [np.mean(paradigm_dict[participant]['audiogram']['estimation'])] * nrows
             })

        paradigm_df = pd.concat([paradigm_df, participant_df])

    return paradigm_df


def downsample_to_3AFC(data_3AFC, data_continuous, frequencies_3AFC, frequencies_100):
    for (paradigm, pred, participant), participant_audiogram in \
            data_continuous.groupby(["paradigm", "pred", "participant"]):

        thresholds_3AFC = []
        thresholds_100 = participant_audiogram.thresholds

        for freq in frequencies_3AFC:
            # Use the data directly if there is a threshold for that 3AFC frequency
            if freq in frequencies_100:
                thresholds_3AFC.append(
                    participant_audiogram.loc[participant_audiogram.frequencies == freq, 'thresholds'].iloc[0])
            # Otherwise interpolate using the closest values
            else:

                thresholds_3AFC.append(interp(freq, frequencies_100, thresholds_100))

        audiogram_df = pd.DataFrame(
            {'participant': [participant] * len(frequencies_3AFC),
             'paradigm': [paradigm] * len(frequencies_3AFC),
             'pred': [pred] * len(frequencies_3AFC),
             'time_pred': participant_audiogram.time_pred[:len(frequencies_3AFC)],
             'freq_pred': participant_audiogram.freq_pred[:len(frequencies_3AFC)],
             'tested_frequencies': [participant_audiogram.tested_frequencies[0] for _ in frequencies_3AFC],
             'tested_levels': [participant_audiogram.tested_levels[0] for _ in frequencies_3AFC],
             'responses': [participant_audiogram.responses[0] for _ in frequencies_3AFC],
             'len_init': participant_audiogram.len_init[:len(frequencies_3AFC)],
             'thresholds': thresholds_3AFC,
             'frequencies': frequencies_3AFC,
             'mean_threshold': [np.mean(thresholds_3AFC)] * len(frequencies_3AFC),
             'nReversals': [data_3AFC.nReversals.iloc[0]] * len(frequencies_3AFC)
             }
        )

        data_3AFC = pd.concat([data_3AFC, audiogram_df])

    return data_3AFC


def compute_db_gain(df, variable):

    precision = StatsParams.pval_precision

    # Group by participant and paradigm, then calculate the mean threshold
    avg_thresholds = df.groupby(['participant', 'paradigm'])[variable].mean().reset_index()

    # Pivot table to have paradigms as columns
    pivoted_df = avg_thresholds.pivot(index='participant', columns='paradigm', values='thresholds')

    # Compute the difference
    pivoted_df['diff'] = pivoted_df['Bayesian'] - pivoted_df['3AFC']

    print("Mean:", round(pivoted_df['diff'].mean(), precision),
          "dB  |  SD:", round(pivoted_df['diff'].std(), precision),
          "dB  |  SEM:", round(pivoted_df['diff'].sem(), precision),
          "dB")
