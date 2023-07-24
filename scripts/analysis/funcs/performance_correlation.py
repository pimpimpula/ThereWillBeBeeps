import pandas as pd
from scipy.stats import pearsonr

from scripts.stats import StatsParams
from scripts.utils import *


def prepare_corr_data(df):
    """
    Translates the values in the 'pred' column using the `translate_conditions` function,
    groups the DataFrame by the 'participant' column, and sorts the groups based on the 'pred' and 'paradigm' columns.
    It then constructs a new DataFrame `matrix_for_corr` with the values of either the 'distance_p50' or 'mean_threshold'
    column from each group, with the 'paradigm' and 'pred' values as the index.

    Parameters:
        df (pd.DataFrame):
            The input DataFrame. Expected to contain 'participant', 'pred', 'paradigm',
            and either 'distance_p50' or 'mean_threshold' columns.

    Returns:
        matrix_for_corr (pd.DataFrame):
            The transformed data ready for correlation analysis.
        df (pd.DataFrame):
            The modified input DataFrame after translating the 'pred' column and grouping/sorting operations.
    """

    # Apply translate_conditions to 'pred' column
    df['pred'] = df['pred'].apply(lambda x: translate_conditions(x) if x in ['none', 'time', 'frequency', 'both'] else x)

    matrix_for_corr = pd.DataFrame()

    # Dictionary to map sort keys
    sort_key_pred = {"R": 1, "T": 2, "F": 3, "FT": 4}
    sort_key_paradigm = {"Bayesian": 1, "Continuous": 2, "Cluster": 3, "3AFC": 4}

    # Grouping by participant and sorting the values for each participant
    for participant, participant_group in df.groupby('participant'):

        participant_group = participant_group.sort_values(by='pred', key=lambda x: x.map(sort_key_pred))
        participant_group = participant_group.sort_values(by='paradigm', key=lambda x: x.map(sort_key_paradigm))

        # Automatically detect the variable of interest
        column_name = 'distance_p50' if 'distance_p50' in df.columns else 'mean_threshold'

        matrix_for_corr.insert(len(matrix_for_corr.columns), participant, participant_group[column_name].to_list())

    matrix_for_corr.index = participant_group[['paradigm', 'pred']]
    matrix_for_corr = matrix_for_corr.transpose()

    return matrix_for_corr


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


def compute_correlation_matrix(df):
    """
    Compute the correlation matrix and the corresponding p-values.

    Parameters:
        df (DataFrame): Input DataFrame for which to compute the correlation matrix and p-values.

    Returns:
        Tuple[DataFrame, DataFrame]: Correlation matrix and p-value matrix.
    """

    # Compute the correlation matrix
    corr = df.corr(method='pearson')
    # Set the upper triangle of the correlation matrix to NaN
    corr = corr.where(~np.triu(np.ones(corr.shape)).astype('bool'))

    # Compute the p-values
    p_values = df.corr(method=pearsonr_pval)
    # Set the upper triangle of the p-value matrix to NaN
    p_values = p_values.where(~np.triu(np.ones(corr.shape)).astype('bool'))

    return corr, p_values


def print_sig_corr(corr_matrix, pval_matrix):
    # Define a list to hold the dictionary results
    corr_rows = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):  # Loop over lower triangle
            condition_1 = corr_matrix.columns[i]
            condition_2 = corr_matrix.columns[j]
            p_value = pval_matrix.iloc[i, j]
            R = corr_matrix.iloc[i, j]

            # Append this data to the list
            corr_rows.append({
                'Condition_1': condition_1,
                'Condition_2': condition_2,
                'PearsonR': R,
                'p-value': p_value,
                'sig': '*' if p_value < StatsParams.alpha else ''
            })

    # Convert the list of dicts to a DataFrame
    corr_results = pd.DataFrame(corr_rows)

    # Remove pairs with p-value > alpha
    corr_results_significant = corr_results[corr_results['p-value'] < StatsParams.alpha]

    # Print DataFrame
    print(corr_results_significant.round(3))


def difference_with_random(df):
    # Create a new DataFrame for the differences
    diff_df = pd.DataFrame()

    for paradigm in ['Continuous', 'Cluster']:
        diff_df[(paradigm, 'R-T')] = df[(paradigm, 'R')] - df[('Continuous', 'T')]
        diff_df[(paradigm, 'R-F')] = df[(paradigm, 'R')] - df[(paradigm, 'F')]
        diff_df[(paradigm, 'R-FT')] = df[(paradigm, 'R')] - df[(paradigm, 'FT')]

    return diff_df
