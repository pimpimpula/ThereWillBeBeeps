import pingouin as pg

from scripts.stats import StatsParams, StatsFormatter
from scripts.utils import *


class P50DataAnalyzer:
    def __init__(self, sigmoid_data):
        self.sigmoid_data = sigmoid_data

    def remove_problematic_participants(self, filter_on, conditions):

        filtered_data = self.sigmoid_data.loc[
            self.sigmoid_data[filter_on].isin(conditions) if filter_on else self.sigmoid_data,
            ['participant', 'paradigm', 'pred', 'distance_p50']].drop_duplicates()

        # Find problematic participants for this analysis
        pb_participants = filtered_data.loc[filtered_data.distance_p50.isna(), 'participant'].unique()

        print(f"\nNumber of participants with at least one missing p50 value (prevent running repeated measures anova):",
              len(pb_participants))

        # Remove problematic participants
        filtered_data = filtered_data[~filtered_data.participant.isin(pb_participants)]

        print("\nRemaining participants after removal of problematic: N =", len(filtered_data.participant.unique()), '\n')

        return filtered_data

    @staticmethod
    def print_descriptive_stats(df):

        precision = StatsParams.pval_precision

        for paradigm, paradigm_data in df.groupby('paradigm'):

            for pred, pred_group in paradigm_data.groupby('pred'):

                prediction = pred if paradigm == pred else translate_conditions(pred)

                print(f"------------------------ {paradigm if paradigm == pred else paradigm + '/' + prediction} ------------------------")
                print("Mean:", round(paradigm_data.loc[paradigm_data['pred'] == pred].distance_p50.mean(), precision),
                      "dB  |  SD:", round(paradigm_data.loc[paradigm_data['pred'] == pred].distance_p50.std(), precision),
                      "dB  |  SEM:", round(paradigm_data.loc[paradigm_data['pred'] == pred].distance_p50.sem(), precision),
                      "dB")

            if len(df.paradigm.unique()) == 2:
                print("")
        print("")

    def p50_stats_pipeline(self, var: str, factor: str, filter_on=False, conditions: list = None):

        # filter data & remove problematic participants for this analysis
        filtered_data = self.remove_problematic_participants(filter_on, conditions)

        # Print descriptive stats
        self.print_descriptive_stats(filtered_data)

        # run one-way ANOVA
        aov_results = pg.rm_anova(data=filtered_data, dv=var,
                                  within=factor, subject='participant')  # , detailed=True)

        StatsFormatter.print_1way_anova(StatsFormatter, aov_results, var=var, factor=factor)

        # post-hoc paired T-tests
        pairwise_results = pg.pairwise_tests(data=filtered_data,
                                             dv=var,
                                             within=factor,
                                             subject='participant',
                                             alternative='two-sided',
                                             padjust=StatsParams.mult_comp)

        if factor == 'pred':
            # Translate predictability to article labels
            pairwise_results['A'] = pairwise_results['A'].apply(translate_conditions)
            pairwise_results['B'] = pairwise_results['B'].apply(translate_conditions)

        # Display post-hoc results
        StatsFormatter.print_paired_ttest_posthocs(StatsFormatter, pairwise_results)

        if conditions == ['Continuous', 'Cluster']:
            aov2_results = pg.rm_anova(data=filtered_data, dv=var,
                                      within=[filter_on, factor], subject='participant')  # , detailed=True)

            StatsFormatter.print_2way_anova(StatsFormatter, aov2_results,
                                            var='p50s' if var == 'distance_p50' else '',  #TODO: fix if reused
                                            factors=[filter_on, factor])
            return filtered_data, aov_results, aov2_results, pairwise_results
        else:
            return filtered_data, aov_results, pairwise_results