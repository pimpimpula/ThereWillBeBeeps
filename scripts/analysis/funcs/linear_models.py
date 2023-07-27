import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scripts.utils import translate_conditions


class RSquared:

    @staticmethod
    def fit_model(x, y):
        x_lm = sm.add_constant(x)
        model = sm.OLS(y, x_lm)
        results = model.fit()
        return results.rsquared, results.pvalues[1]

    def compute_r_squared(self, paradigm, data, xvar, yvar):

        if len(data.pred.unique()) == 4:
            preds = ['none', 'time', 'frequency', 'both']
        else:
            preds = ['time', 'frequency', 'both']

        R_values = {pred: self.fit_model(data.loc[(data.paradigm == paradigm) & (data.pred == pred)][xvar].to_numpy(),
                                         data.loc[(data.paradigm == paradigm) & (data.pred == pred)][yvar].to_numpy())
                    for pred in preds}
        return R_values

    def fit_linear_model_to_chosen_data(self, data, xvar, yvar):
        paradigms = ['Continuous', 'Cluster']
        list_of_dicts = []
        for paradigm in paradigms:
            print("\n---" + paradigm.upper() + "---")
            r_values = self.compute_r_squared(paradigm, data, xvar, yvar)
            for pred, (r_value, p_value) in r_values.items():
                print(f"{translate_conditions(pred):<2}: R2: {np.round(r_value, 3):<5}, p {'< 0.001' if p_value < 0.001 else '= ' + str(np.round(p_value, 3)):<9} {'*' if p_value < 0.05 else ''}")

                list_of_dicts.append({'paradigm': paradigm, 'pred': pred, 'R2': r_value, 'p_value': p_value})
        df_r_values = pd.DataFrame(list_of_dicts)
        return df_r_values


def compute_difference_w_random(all_data):
    data_diff = copy.deepcopy(all_data)
    data_diff['threshold_diff'] = float('NaN')
    data_diff['p50_diff'] = float('NaN')

    # Loop over the groups based on 'paradigm' and 'participant'
    for (paradigm, participant), group in all_data.groupby(['paradigm', 'participant']):

        # Get the values for the 'none' condition
        none_condition = group[group['pred'] == 'none']

        if not none_condition.empty:
            # Calculate the difference and update the respective rows in data_diff
            for index, row in group.iterrows():
                if row['pred'] != 'none':
                    data_diff.at[index, 'threshold_diff'] = row['mean_threshold'] - none_condition['mean_threshold'].values[0]
                    data_diff.at[index, 'p50_diff'] = row['distance_p50'] - none_condition['distance_p50'].values[0]

    data_diff = data_diff[data_diff.pred != 'none']

    return data_diff


def remove_nan_participants(data):
    data_noNan = copy.deepcopy(data)

    for paradigm, group in data.groupby('paradigm'):
        nan_participants = group.loc[group.distance_p50.isnull(), 'participant'].unique()
        data_noNan = data_noNan.drop(data_noNan.loc[(data_noNan.paradigm == paradigm) & data_noNan.participant.isin(nan_participants)].index)

    return data_noNan
