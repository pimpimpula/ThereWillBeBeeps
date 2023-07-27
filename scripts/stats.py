import pandas as pd


class StatsParams:
    alpha = 0.05  # threshold for significance
    mult_comp = 'fdr_bh'  # type of correction for multiple comparison
    pval_precision = 3


class StatsFormatter:

    @staticmethod
    def fix0(pval):
        return "<0.001" if pval < 0.001 else f"{pval:.{StatsParams.pval_precision}f}"

    @staticmethod
    def print_1way_anova(self, aov_results: pd.DataFrame, var: str, factor: str):
        print(f"One-way ANOVA on the {var} (factor: {factor}):")
        for _, row in aov_results.iterrows():
            # Use Greenhouse-Greenhouse-Geisser correction if available
            # pval = row['p-GG-corr'] if 'p-GG-corr' in row.keys() else row['p-unc']
            pval = row['p-unc']
            significance = '     *' if pval < StatsParams.alpha else ''
            print(
                f"{row['Source']}: F({int(row['ddof1'])}, {int(row['ddof2'])}) = {row['F']:.2f}, p = {self.fix0(pval)}{significance}")

        print("")

    @staticmethod
    def print_2way_anova(self, aov_results: pd.DataFrame, var: str, factors):
        print(f"Two-way ANOVA ({', '.join(factors)}) on the {var}: \n")
        print("Source                   F        ddof1  ddof2   p-unc    sig")
        for _, row in aov_results.iterrows():
            # print(row['p-GG-corr'])
            # pval = row['p-GG-corr'] if 'p-GG-corr' in row.keys() else row['p-unc']
            pval = row['p-unc']
            significance = '*' if pval < StatsParams.alpha else ''

            print("{:<25}{:<10.2f}{:<7}{:<7}{:<10}{}".format(
                row['Source'], row['F'], int(row['ddof1']), int(row['ddof2']), self.fix0(row['p-unc']), significance))

        print("")

    @staticmethod
    def print_paired_ttest(self, pairwise_results: pd.DataFrame, var: str):
        print(f"{var} comparison (paired T-test):")
        for _, row in pairwise_results.iterrows():
            significance = '     *' if row['p-unc'] < StatsParams.alpha else ''
            print(
                f"{row['A']} vs {row['B']}: T({int(row['dof'])}) = {row['T']:.2f}, p = {self.fix0(row['p-unc'])}{significance}")

        print("")

    def print_paired_ttest_posthocs(self, pairwise_results: pd.DataFrame):
        print("Post-hoc tests (paired t-tests):")

        for _, row in pairwise_results.iterrows():
            significance = '     *' if row['p-corr'] < StatsParams.alpha else ""
            print(
                f"{row['Contrast']} / {row['A']} vs {row['B']}: T({int(row['dof'])}) = {row['T']:.2f}, p = {self.fix0(row['p-unc'])} ({StatsParams.mult_comp}: {self.fix0(row['p-corr'])}){significance}")

        print("")
