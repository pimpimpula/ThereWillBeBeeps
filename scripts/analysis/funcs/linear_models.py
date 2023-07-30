import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scripts.utils import translate_conditions


class RSquared:

    @staticmethod
    def fit_model(x, y):
        """
        Fit an Ordinary Least Squares (OLS) linear regression model to the data.

        Parameters
        ----------
        x : array-like
            Input data (predictor variable). This can be a 1-D array (for a single predictor) or a 2-D array
            (for multiple predictors).

        y : array-like
            Response variable. Should be a 1-D array of the same length as `x`.

        Returns
        -------
        tuple
            A tuple containing two values:

                R-squared value - Proportion of the variance for the dependent variable that's explained
                by the independent variable(s).

                p-value of the predictor(s) - Probability that the null hypothesis (predictor has no effect) is true.

        Notes
        -----
        The function uses the `OLS` method from the `statsmodels` library to fit the model. It adds a constant term
        to the predictor data to represent the intercept.
        """

        x_lm = sm.add_constant(x)
        model = sm.OLS(y, x_lm)
        results = model.fit()
        return results.rsquared, results.pvalues[1]

    def compute_r_squared(self, paradigm, data, xvar, yvar):
        """
        This function calculates the R-squared values for a set of prediction conditions.
        It fits a model for each prediction condition within a specific paradigm, and returns the R-squared value for each model.

        Args:
            paradigm (str): The paradigm under consideration ('Continuous' or 'Cluster').
            data (pandas.DataFrame): The data frame containing the data.
            xvar (str): The name of the independent variable in the data.
            yvar (str): The name of the dependent variable in the data.

        Returns:
            dict: A dictionary where keys are prediction conditions and values are tuples, with the first element being the
                  R-squared value and the second being the p-value of the fitted model.
        """

        if len(data.pred.unique()) == 4:
            preds = ['none', 'time', 'frequency', 'both']
        else:
            preds = ['time', 'frequency', 'both']

        R_values = {pred: self.fit_model(data.loc[(data.paradigm == paradigm) & (data.pred == pred)][xvar].to_numpy(),
                                         data.loc[(data.paradigm == paradigm) & (data.pred == pred)][yvar].to_numpy())
                    for pred in preds}
        return R_values

    def compute_and_report_r_squared(self, data, xvar, yvar):
        """
        Compute and report the R-squared and p-values grouped by predictability conditions, for each paradigm.

        Parameters
        ----------
        self : object instance

        data : pandas.DataFrame
            The DataFrame containing the data.

        xvar : str
            The name of the variable in 'data' to be used as the predictor in the regression.

        yvar : str
            The name of the variable in 'data' to be used as the response in the regression.

        Returns
        -------
        df_r_values : pandas.DataFrame
            A DataFrame containing the calculated R-squared and p-values for each condition in each paradigm. Each row corresponds to a condition within a paradigm. Columns include 'paradigm', 'pred', 'R2', and 'p_value'.

        Notes
        -----
        This function operates on two paradigms: 'Continuous' and 'Cluster'. It computes the R-squared and p-values for each condition within these paradigms, and prints these values. Conditions with a p-value less than 0.05 are indicated with an asterisk.
        """

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
    """
    This function computes the difference in threshold and p50 values relative
    to the 'none' condition for each participant within each paradigm.

    Args:
        all_data (pandas.DataFrame): The dataframe containing the data. It should have
                                     columns 'paradigm', 'participant', 'pred',
                                     'mean_threshold', and 'distance_p50'.

    Returns:
        pandas.DataFrame: A dataframe similar to the input but with two additional columns
                          'threshold_diff' and 'p50_diff' representing the difference in
                          threshold and p50 values relative to the 'none' condition, respectively.
                          Rows where 'pred' is 'none' are removed.
    """

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
    """
    This function removes participants with NaN values in the 'distance_p50' column for each paradigm.

    Args:
    data (pandas.DataFrame): The dataframe containing the data. It should have columns 'paradigm', 'participant', and 'distance_p50'.

    Returns:
    pandas.DataFrame: A dataframe similar to the input but with participants who have NaN values in 'distance_p50' removed.
    """

    data_noNan = copy.deepcopy(data)

    for paradigm, group in data.groupby('paradigm'):
        nan_participants = group.loc[group.distance_p50.isnull(), 'participant'].unique()
        data_noNan = data_noNan.drop(data_noNan.loc[(data_noNan.paradigm == paradigm) & data_noNan.participant.isin(nan_participants)].index)

    return data_noNan


class LMM:

    def __init__(self):
        pass

