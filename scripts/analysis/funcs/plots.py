import math

import numpy as np
from matplotlib import gridspec
from pandas import DataFrame

import matplotlib.patches as mpatches

from scripts.figure_params import *
from scripts.stats import StatsParams
from scripts.utils import translate_conditions


class Fig3Thresholds:

    def __init__(self):
        pass

    @staticmethod
    def plot_mean_audiogram(freqs, thresholds, error, color, label):
        # print(thresholds, error)
        plt.fill_between(freqs,
                         thresholds - error,
                         thresholds + error,
                         color=color, zorder=1, alpha=0.3, linewidth=0)
        plt.plot(freqs, thresholds, label=label, color=color, linewidth=3, zorder=3)

    def plot_3afc_vs_randomized_audiogram(self, audiograms_3afc, audiograms_100, frequencies_3afc, frequencies_100):
        update_plot_params()

        audiogram_3afc = audiograms_3afc.groupby('frequencies').thresholds.mean()
        error_3afc = audiograms_3afc.groupby('frequencies').thresholds.sem()

        audiogram_bayes = audiograms_100.groupby('frequencies').thresholds.mean()
        error_bayes = audiograms_100.groupby('frequencies').thresholds.sem()

        # Add vertical lines for the frequencies tested in the 3-AFC
        plt.vlines(frequencies_3afc, -11.5, 9, zorder=0, ls=':', color='k', alpha=.5)

        # Plot audiograms
        for paradigm, label, freqs, thresholds, error, in zip(['3AFC', 'Bayesian'],
                                                              ['3-AFC', 'Randomized'],
                                                              [frequencies_3afc, frequencies_100],
                                                              [audiogram_3afc, audiogram_bayes],
                                                              [error_3afc, error_bayes]):
            self.plot_mean_audiogram(freqs, thresholds, error, paradigms_palette(paradigm), label)

        plt.xscale('log')
        plt.xticks(frequencies_3afc, [125, 250, 500, 750, '1K', '1.5K', '2K', '3K', '4K', '6K', '8K'])
        plt.ylim([10, -12.5])
        plt.minorticks_off()
        # ax = plt.gca()
        # ax.invert_yaxis()
        plt.title(f'Average audiogram for the Randomized and 3-AFC tasks',
                  fontdict={'family': fig_params().title_font})
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Threshold (in dB HL)')
        plt.legend(loc='upper left')  # fontsize='x-small'

    @staticmethod
    def barplot_3afc_vs_randomized(audiograms_3afc, metric):
        update_plot_params()

        # Compute threshold and SEM
        threshold_bayes = audiograms_3afc.loc[audiograms_3afc.paradigm == 'Bayesian'].groupby(
            'participant').thresholds.mean().mean()
        error_bayes = audiograms_3afc.loc[audiograms_3afc.paradigm == 'Bayesian'].groupby(
            'participant').thresholds.mean().sem()
        threshold_3afc = audiograms_3afc.loc[audiograms_3afc.paradigm == '3AFC'].groupby(
            'participant').thresholds.mean().mean()
        error_3afc = audiograms_3afc.loc[audiograms_3afc.paradigm == '3AFC'].groupby('participant').thresholds.mean().sem()

        plt.hlines(0, 0.25, 2.75, zorder=0, color='k', alpha=.5)

        # Barplot
        plt.bar(1, threshold_bayes, color=paradigms_palette('Bayesian'), width=.5)
        plt.errorbar(1, threshold_bayes, yerr=error_bayes,
                     color='black', linewidth=3)  # , capsize=6)
        plt.bar(2, threshold_3afc, color=paradigms_palette('3AFC'), width=.5)
        plt.errorbar(2, threshold_3afc, yerr=error_3afc,
                     color='black', linewidth=3)  # , capsize=6)

        # Individual data
        for participant, participant_data in audiograms_3afc.groupby('participant'):
            participant_jitter = np.random.uniform(-0.1, 0.1)
            x = np.array([2, 1]) + participant_jitter

            plt.scatter(x, participant_data.groupby('paradigm').thresholds.mean(),
                        s=8, edgecolors='black', facecolors='none', alpha=.2)

            plt.plot(x, participant_data.groupby('paradigm').thresholds.mean(),
                     'black', alpha=.2)

        # ax = plt.gca()
        # ax.invert_yaxis()
        # plt.ylim([10, -15])
        plt.xlim(0.25, 2.75)
        plt.xticks([1, 2], ['Randomized', '    3-AFC'])
        plt.title(f'Average {metric}', fontdict={"family": fig_params().title_font})
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel(f"Mean {metric} (in dB HL)")


class Fig2CMethods:

    def __init__(self, audiogram_data: DataFrame, sigmoid_data: DataFrame, trials_data: DataFrame,
                 chosen_paradigm: str, chosen_pred: str):

        self.random_audiogram = audiogram_data
        self.sigmoid_data = sigmoid_data
        self.trials_data = trials_data
        self.paradigm = chosen_paradigm
        self.chosen_pred = chosen_pred
        self.pred = translate_conditions(chosen_pred) if chosen_paradigm in ['Continuous', 'Cluster'] else chosen_pred

    def plot_example_global_random_audiogram(self):
        """
        Plots the global random audiogram data for a chosen participant, with tested tones color-coded by task.

        Returns:
            fig (matplotlib.figure.Figure): The output figure.
        """

        update_plot_params()

        audiogram_thresholds = self.random_audiogram.random_thresholds[0]
        audiogram_freqs = self.random_audiogram.frequencies[0]

        tested_freqs = self.random_audiogram.random_tested_frequencies[0]
        tested_levels = self.random_audiogram.random_tested_levels[0]
        responses = self.random_audiogram.random_responses[0]

        # for positive detections as green checkmarks & misses as red crosses
        # color_map = ['green' if resp == 1 else 'red' for resp in responses]

        # for trials color-coded by paradigms
        ntones = self.random_audiogram.random_ntones[0]
        paradigms, ns = np.array(ntones).T
        colors = [get_color(paradigm) for paradigm in paradigms]
        colors[0] = '#1E2023'  # darken Randomized color to improve contrast for this plot
        color_map = [color for color, n in zip(colors, ns) for _ in range(int(n))]

        # markers
        marker_map = {1: u'$\u2713$', 0: 'X'}
        size_map = {1: 400, 0: 100}

        #######################
        #    Create figure    #
        #######################

        fig = plt.figure(figsize=[9, 4.5])

        # plot random audiogram
        plt.plot(audiogram_freqs, audiogram_thresholds,
                 c='k', lw=2.5, zorder=0,
                 label='random audiogram')

        # plot individual trials
        for x, y, resp, color in zip(tested_freqs, tested_levels, responses, color_map):
            plt.scatter(x, y, marker=marker_map[resp], c=color, s=size_map[resp], edgecolors='white', lw=.5, alpha=1)

        # aesthetics
        plt.xscale('log')
        plt.xticks([125, 1000, 8000], [125, 1000, 8000])
        plt.minorticks_off()
        ax = plt.gca()
        ax.invert_yaxis()

        # Create custom legend elements, bit ugly
        # legend_elements = []
        # for paradigm, color in zip(paradigms, colors):
        #     legend_label = paradigm + "\n" + u'$\u2713$' + ": Detected\n" + 'X' + ": Missed"
        #     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=legend_label))
        # plt.legend(handles=legend_elements)

        plt.legend(loc='lower left')

        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Level (in dB HL)')
        plt.title('Global random audiogram')

        plt.tight_layout()

        return fig

    def plot_example_distances(self):
        """
        Plots the audiogram data for a specific participant along with trials for a specific paradigm and pred condition.

        Returns:
            fig (matplotlib.figure.Figure): The output figure.
        """
        update_plot_params()

        fig = plt.figure(figsize=fig_params()['2C_dims'])

        # Plot the global random audiogram
        audiogram_thresholds = self.random_audiogram.random_thresholds[0]
        audiogram_freqs = self.random_audiogram.frequencies[0]
        plt.plot(audiogram_freqs, audiogram_thresholds, c='k', lw=2.5, zorder=5, label='random audiogram')

        # Create the color and marker maps
        color_map = ['green' if resp == 1 else 'red' for resp in self.trials_data.responses]
        marker_map = {1: u'$\u2713$', 0: 'X'}
        size_map = {1: 400, 0: 100}  # adjust sizes here

        # Plot the trials
        for idx, (freq, level, resp, random_threshold) in enumerate(zip(self.trials_data.tested_frequencies,
                                                                        self.trials_data.tested_levels,
                                                                        self.trials_data.responses,
                                                                        self.trials_data.random_threshold)):
            plt.scatter(freq, level,
                        c=color_map[idx], marker=marker_map[resp], s=size_map[resp],
                        edgecolors='white', lw=.5, alpha=1)
            plt.vlines(x=freq, ymin=level, ymax=random_threshold,
                       # color=color_map[idx], linestyles=':',
                       color='grey',
                       alpha=0.7, zorder=0)

        # Aesthetics
        plt.xscale('log')
        plt.xticks([125, 1000, 8000], [125, 1000, 8000])
        plt.ylim([20, -20])
        plt.minorticks_off()
        # ax = plt.gca()
        # ax.invert_yaxis()
        plt.legend(loc='lower left')
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Level (in dB HL)')
        plt.title(f"Tone distance from threshold (example data from {self.paradigm}{'' if self.paradigm in ['3AFC', 'Bayesian'] else f'/{self.pred}'})")

        plt.tight_layout()

        return fig

    def plot_example_sigmoid(self):

        update_plot_params()

        color_map = ['green' if resp == 1 else 'red' for resp in self.trials_data.responses]
        # markers
        marker_map = [u'$\u2713$' if resp == 1 else 'X' for resp in self.trials_data.responses]
        size_map = self.trials_data['responses'].replace({1: 400, 0: 100})

        fig = plt.figure(figsize=fig_params()['2C_dims'])

        plt.hlines(.5, self.sigmoid_data.random_distance.min(), self.sigmoid_data.random_distance.max(),
                   colors='grey', zorder=1)
        plt.plot(self.sigmoid_data.random_distance, self.sigmoid_data.sigmoid_probas, c='k', zorder=3)
        plt.scatter(self.sigmoid_data.distance_p50.unique()[0], .5, s=50, c='k', zorder=4, label='p50')

        # Plot each point individually with its own marker
        for i, (x, y, m, s, c) in enumerate(zip(self.trials_data.random_distance,
                                                self.trials_data.responses,
                                                marker_map,
                                                size_map,
                                                color_map)):
            plt.scatter(x, y, marker=m, s=s, facecolors=c, edgecolors='white', lw=.75, alpha=1)


        plt.yticks([0, .5, 1], ["0", "0.5", "1"])
        xlim_min = 5 * math.floor(self.trials_data.random_distance.min() / 5)
        xlim_max = 5 * math.ceil(self.trials_data.random_distance.max() / 5)
        plt.xlim([xlim_min, xlim_max])

        plt.legend()

        plt.title(f"p50 (example data from {self.paradigm}{'' if self.paradigm in ['3AFC', 'Bayesian'] else f'/{self.pred}'})")
        plt.xlabel("Distance from random threshold (dB HL)")
        plt.ylabel("Proportion detected")

        plt.tight_layout()

        return fig


class VisualChecksP50:

    @staticmethod
    def plot_problematic_p50s(ax, sigmoid, xsigmoid, p50, participant_group, anomaly_type, problematic_participants):

        participant = participant_group.participant.unique()[0]
        paradigm = participant_group.paradigm.unique()[0]
        pred = translate_conditions(participant_group.pred.unique()[0])

        if participant in problematic_participants:
            ax.set_facecolor('#ECECEC')

        ax.fill_betweenx([-0.1, 1.1],
                         participant_group.random_distance.min(), participant_group.random_distance.max(),
                         facecolors='r' if anomaly_type == 'p50 outside range' else 'grey', alpha=.1, zorder=0)
        ax.hlines(.5, min(xsigmoid), max(xsigmoid),
                  colors='grey', zorder=1)
        ax.scatter(participant_group.random_distance, participant_group.responses,
                   s=20, c='k', alpha=.3)
        ax.plot(xsigmoid, sigmoid, c='r' if anomaly_type == 'Inverted sigmoid' else 'k', zorder=3)
        ax.scatter(p50, .5, s=50, c='r' if anomaly_type == 'p50 outside range' else 'k', zorder=4)

        ax.set_title(f"{paradigm} ({pred}) - {participant.upper()}")
        ax.set_xlabel("Distance from random thresold (dB HL)")
        ax.set_yticks([0, .5, 1])

        plt.tight_layout()

    @staticmethod
    def plot_all_sigmoids(pseudo_psychometric_curves):

        paradigm_location = {'Bayesian': 4, 'Cluster': 9, 'Continuous': 5, '3AFC': 13}
        pred_location = {'both': 0, 'frequency': 1, 'time': 2, 'none': 3}

        update_plot_params()

        fig = plt.figure(figsize=(15, 12))

        # Loop over paradigms
        for paradigm, paradigm_group in pseudo_psychometric_curves.groupby('paradigm'):

            problematic_participants= paradigm_group[paradigm_group.problematic].participant.unique()

            paradigm_group = paradigm_group[~paradigm_group.participant.isin(problematic_participants)]

            xdistance = list(np.linspace(paradigm_group.random_distance.min(),
                                         paradigm_group.random_distance.max(), num=100))

            # Loop through each prediction condition within the current paradigm
            for pred, pred_group in paradigm_group.groupby('pred'):

                # Translate pred condition to article labels
                if paradigm in ['Continuous', 'Cluster']:
                    prediction = translate_conditions(pred)

                color = get_color(paradigm, pred)

                # Subplot arrangement
                if paradigm == '3AFC':
                    ax = plt.subplot(4, 4, (paradigm_location[paradigm]))
                else:
                    plt.subplot(4, 4, (paradigm_location[paradigm] + pred_location[pred]
                                       if paradigm in ['Cluster', 'Continuous']
                                       else paradigm_location[paradigm]),
                                sharex=ax, sharey=ax)

                plt.hlines(0.5, np.min(xdistance), np.max(xdistance), colors='black', lw=1, alpha=.3, zorder=1)
                plt.vlines(0, 0, 1, colors='black', lw=1, alpha=.3, zorder=0)

                # Loop through each participant within the current prediction condition
                for participant, participant_group in pred_group.groupby('participant'):
                    # Plot participant sigmoid and p50 value if not outlier
                    plt.plot(xdistance, participant_group.sigmoid_probas, 'black', alpha=.1, lw=1, zorder=3)
                    plt.scatter(participant_group.distance_p50.iloc[0], 0.5, c='black', alpha=.1, zorder=3, s=2)

                # Doesn't work now that we're looping over psychometric_curves_individual rather than trials_data
                # I'm too lazy to fix it since we're not using the figure
                # # Plot all answers for this pred group
                # plt.scatter(pred_group.loc[pred_group.outlier == False, distance],
                #             pred_group.loc[pred_group.outlier == False, 'yi'] + np.random.uniform(-0.05, 0.05, len(
                #                 pred_group.loc[pred_group.outlier == False])),
                #             edgecolors='grey', facecolors='None', alpha=.1, s=2,
                #             label="Actual data", zorder=2)

                # Plot the mean p50 value for the current prediction condition (across participants)
                plt.scatter(pred_group[['participant', 'distance_p50']].drop_duplicates().distance_p50.mean(), .5,
                            facecolors='black', edgecolors='None', s=30,
                            label="", zorder=5)

                # Plot the average sigmoid and SEM for each condition
                P50GlobalFigure.draw_sigmoid_and_error(
                    P50GlobalFigure.calculate_mean_sem(pred_group), color, pred, paradigm, zorders=[4, 3.5])

                if paradigm in ['Bayesian', '3AFC']:
                    plt.title(paradigm)
                else:
                    plt.title(f"{paradigm} / {prediction}")

                plt.xlabel(f"Distance from global threshold (dB HL)")
                plt.ylabel("Proportion detected")
                plt.xticks()
                plt.yticks()
                plt.ylim([-0.1, 1.1])
                # plt.xlim([np.min(xdistance), np.max(xdistance)])

        plt.tight_layout()


class P50GlobalFigure:

    def __init__(self, sigmoid_data):
        self.sigmoid_data = sigmoid_data
        self.Pn = {"Bayesian": 1, "Continuous": 3.5, "Cluster": 8.5, "3AFC": 13.5}
        self.pn = {'none': 0, 'time': 1, 'frequency': 2, 'both': 3}
        self.legend_order = ['FT', 'F', 'T', 'R']
        update_plot_params()

    @staticmethod
    def calculate_mean_sem(df):
        # Group by 'random_distance' and calculate mean and SEM for 'sigmoid_probas'
        grouped = df.groupby('random_distance')['sigmoid_probas'].agg(['mean', 'sem']).reset_index()
        return grouped

    @staticmethod
    def draw_sigmoid_and_error(data, color, pred, paradigm, zorders):
        plt.fill_between(data.random_distance,
                         data['mean'] - data['sem'],
                         data['mean'] + data['sem'],
                         color=color, alpha=.5, edgecolor=None,
                         zorder=zorders[0])

        plt.plot(data.random_distance, data['mean'], color,
                 label=translate_conditions(pred) if paradigm in ['Cluster', 'Continuous'] else None,
                 lw=2, zorder=zorders[1])

    def plot_sigmoids_and_barplot(self):

        plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])
        axes = [plt.subplot(gs[i, j]) for i in range(2) for j in range(2)] + [plt.subplot(gs[:, 2])]

        # Specify the order of the legend
        legend_handles = []
        legend_labels = []

        for i, (paradigm, paradigm_group) in enumerate(self.sigmoid_data.groupby('paradigm')):
            paradigm_group = paradigm_group.loc[~paradigm_group.problematic]
            plt.sca(axes[i])
            plt.title(paradigm)
            plt.hlines(0.5, paradigm_group.random_distance.min(), paradigm_group.random_distance.max(),
                       colors='black', lw=1, alpha=.3, zorder=0)
            plt.vlines(0, -0.1, 1.1, colors='black', lw=1, alpha=.3, zorder=0)

            for pred , pred_group in paradigm_group.groupby('pred'):

                color = get_color(paradigm, pred)

                self.draw_sigmoid_and_error(self.calculate_mean_sem(pred_group), color, pred, paradigm,
                                            zorders=[self.pn[pred] + 1 if paradigm in ['Cluster', 'Continuous'] else 2,
                                                     self.pn[pred] + 2 if paradigm in ['Cluster', 'Continuous'] else 3])
                plt.xlabel("Distance from global threshold (dB HL)")
                plt.ylabel("Proportion detected")
                plt.ylim([-0.15, 1.15])
                plt.xlim([-25, 25])

                # Add a handle for the current item to the list of handles
                if paradigm == 'Cluster':
                    current_handle = mpatches.Patch(color=color, label=translate_conditions(pred))
                    legend_handles.append(current_handle)
                    legend_labels.append(translate_conditions(pred))

                current_ax = plt.gca()  # Remember the current axis
                plt.sca(axes[-1])  # Switch to bar plot
                bar_pos = self.Pn[paradigm] + (self.pn[pred] if paradigm in ['Cluster', 'Continuous'] else 0)
                plt.bar(bar_pos, pred_group.distance_p50.mean(), color=color, alpha=1)
                plt.errorbar(bar_pos, pred_group.distance_p50.drop_duplicates().mean(),
                             pred_group.distance_p50.drop_duplicates().sem(),
                             lw=3, color='black', zorder=5)
                plt.sca(current_ax)  # Switch back to the previous plot

        plt.sca(axes[-1])
        plt.hlines(0, 0, 14.5, colors='black', lw=1, alpha=.3, zorder=0)
        plt.title("p50 threshold distance")
        plt.xlabel("Paradigm")
        plt.ylabel("p50 threshold distance (dB HL)")
        plt.xticks(list(self.Pn.values()), self.Pn.keys())
        plt.gca().invert_yaxis()

        # Order the legend according to the specified order
        legend_dict = dict(zip(legend_labels, legend_handles))
        ordered_legend_handles = [legend_dict[label] for label in self.legend_order]

        plt.legend(ordered_legend_handles, self.legend_order)

        plt.tight_layout()


class Fig4p50:

    def __init__(self):
        pass

    @staticmethod
    def p50_barplot(var, paradigms_data, continuous_data, cluster_data):
        fig = plt.figure(figsize=(8, 5.33))

        # Define the grid
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        ylims = [2, -12 if var == 'distance_p50' else -8]

        # Plot paradigm comparison
        ax1 = plt.subplot(gs[0])
        ax1.hlines(0, -0.5, 3.5, 'grey')
        for n, paradigm in enumerate(['Bayesian', 'Continuous', 'Cluster', '3AFC']):
            ax1.bar(n, paradigms_data.loc[paradigms_data.paradigm == paradigm, var].mean(), width=.5,
                    color=get_color(paradigm, 'none' if paradigm in ['Cluster', 'Continuous'] else None))
            ax1.errorbar(n, paradigms_data.loc[paradigms_data.paradigm == paradigm, var].mean(),
                         paradigms_data.loc[paradigms_data.paradigm == paradigm, var].sem(),
                         color='black', elinewidth=3)
        ax1.set_ylim(ylims)
        ax1.set_xlim([-0.5, 3.5])
        ax1.set_ylabel('p50 threshold distance' if var == 'distance_p50' else 'Mean threshold')
        ax1.set_xticks([-0.25, 0.75, 1.75, 2.75])
        ax1.set_xticklabels(['Random', 'Continuous', 'Cluster', '3AFC'], rotation=25)
        ax1.tick_params(axis='x', which='both', length=0)


        # Plot pred comparison
        ax2 = plt.subplot(gs[1])
        ylims = [2, -12] if var == 'distance_p50' else [1, -4]

        ax2.hlines(0, -1, 9, 'grey')
        for n, (paradigm, paradigm_data) in enumerate(zip(['Continuous', 'Cluster'], [continuous_data, cluster_data])):

            for m, pred in enumerate(['none', 'time', 'frequency', 'both']):
                ax2.bar(m + 5 * n,
                        paradigm_data.loc[(paradigm_data.pred == pred), var].mean(),
                        color=pred_palette(pred))
                ax2.errorbar(m + 5 * n, paradigm_data.loc[(paradigm_data.pred == pred), var].mean(),
                             paradigm_data.loc[paradigm_data.pred == pred, var].sem(),
                             color='black', elinewidth=3)
        ax2.set_ylim(ylims)
        ax2.set_xlim([-1, 9])
        ax2.set_xticks([1.5, 6.5])
        ax2.set_xticklabels(['Continuous', 'Cluster'])

        plt.tight_layout()

        return fig


    @staticmethod
    def plot_correlation_subplot(fig, ax, corr, p_values, var):
        """
        Plot a correlation matrix subplot on a given axes.

        Parameters:
            ax (Axes): The axes on which to plot the correlation matrix.
            corr (DataFrame): The correlation matrix.
            p_values (DataFrame): The p-value matrix.
            var (str): The variable ('mean_threshold' or 'distance_p50') to display in the title of the plot.

        Returns:
            None.
        """
        # Get the maximum absolute value of the correlations (rounded up to nearest 0.05)
        cbar_max = np.ceil(corr.abs().max().max() / 0.05) * 0.05

        # Display the correlation matrix
        mappable = ax.imshow(corr, cmap='Spectral_r', vmin=-cbar_max, vmax=cbar_max)

        # Set the x- and y-ticks
        ax.set_yticks(np.arange(len(corr)), ['Randomized' if row[0] in ['Bayesian'] else row[1] for row in corr.index])
        ax.set_xticks(np.arange(len(corr)), ['Randomized' if row[0] in ['Bayesian'] else row[1] for row in corr.index], rotation=40)

        # Add a colorbar
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_yticks([-cbar_max, 0, cbar_max])

        # ax = plt.gca()

        # Loop through each cell in the correlation matrix
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                # If the p-value is less than 0.05, add a scatter plot point
                if p_values.iloc[i, j] < 0.05:
                    ax.scatter(j, i, c='black', s=10, marker='o')

        ax.set_title(f"{'p50' if var == 'distance_p50' else 'Mean threshold'} (Pearson's R)")


    def plot_correlation_matrices(self, corr_threshold, p_values_threshold, corr_p50, p_values_p50):
        """
        Method to plot a correlation matrix of given datasets.

        Parameters:
            corr_threshold (DataFrame): Correlation matrix for the threshold data.
            p_values_threshold (DataFrame): P-value matrix for the threshold data.
            corr_p50 (DataFrame): Correlation matrix for the p50 data.
            p_values_p50 (DataFrame): P-value matrix for the p50 data.

        Returns:
            Figure: A matplotlib Figure instance with the plot of the correlation matrix.
        """

        update_plot_params()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Plot the subplots
        self.plot_correlation_subplot(fig, axs[0], corr_threshold, p_values_threshold, "mean_threshold")
        self.plot_correlation_subplot(fig, axs[1], corr_p50, p_values_p50, "distance_p50")

        fig.tight_layout()

        return fig
