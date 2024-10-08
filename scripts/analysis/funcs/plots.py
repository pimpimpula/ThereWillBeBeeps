import math
from itertools import combinations

import numpy as np
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from pandas import DataFrame

import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

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

    FIG_SIZE = (9, 5)
    LEGEND_SIZE = 0

    def __init__(self, audiogram_data: DataFrame, sigmoid_data: DataFrame, trials_data: DataFrame,
                 chosen_paradigm: str, chosen_pred: str):

        self.random_audiogram = audiogram_data
        self.sigmoid_data = sigmoid_data
        self.trials_data = trials_data
        # self.random_trials_data =
        # self.trials_data =
        self.paradigm = chosen_paradigm
        self.chosen_pred = chosen_pred
        self.pred = translate_conditions(chosen_pred) if chosen_paradigm in ['Continuous', 'Cluster'] else chosen_pred

        update_plot_params()
        plt.rcParams.update(
            {'font.size': 20,
             'axes.titlesize': 20,
             'axes.labelsize': 18,
             'xtick.labelsize': 18,
             'ytick.labelsize': 18,
             'legend.fontsize': 16,
             }
        )

    def sort_trials(self):

        tested_freqs = self.random_audiogram.random_tested_frequencies[0]
        tested_levels = self.random_audiogram.random_tested_levels[0]
        responses = self.random_audiogram.random_responses[0]

        order = ['Bayesian', 'Continuous', 'Cluster']

        ntones = self.random_audiogram.random_ntones[0]

        zipped_trials = list(zip(tested_freqs, tested_levels, responses))

        # Map the paradigm to its start and end index in the data
        mapping = {}
        start_index = 0
        for paradigm, count in ntones:
            mapping[paradigm] = (start_index, start_index + count)
            start_index += count

        ntones = sorted(ntones, key=lambda x: order.index(x[0]))

        sorted_trials = []
        colors = []
        for paradigm, _ in ntones:
            paradigm_trials = zipped_trials[mapping[paradigm][0]:mapping[paradigm][1]]
            sorted_trials.extend(paradigm_trials)
            colors.extend(['#1E2023' if paradigm == 'Bayesian' else get_color(paradigm)] * len(paradigm_trials))

        # Unpack sorted data
        sorted_tested_freqs, sorted_tested_levels, sorted_responses = zip(*sorted_trials)

        return ntones, mapping, sorted_tested_freqs, sorted_tested_levels, sorted_responses, colors

    def plot_example_global_random_audiogram(self):
        """
        Plots the global random audiogram data for a chosen participant, with tested tones color-coded by task.

        Returns:
            fig (matplotlib.figure.Figure): The output figure.
        """

        audiogram_thresholds = self.random_audiogram.random_thresholds[0]
        audiogram_freqs = self.random_audiogram.frequencies[0]

        # for trials color-coded by paradigms
        ntones, mapping, tested_freqs, tested_levels, responses, colors = self.sort_trials()

        # markers
        marker_map = {1: u'$\u2713$', 0: 'X'}
        size_map = {1: 400, 0: 100}

        #######################
        #    Create figure    #
        #######################

        fig = plt.figure(figsize=self.FIG_SIZE)

        # plot random audiogram
        plt.plot(audiogram_freqs, audiogram_thresholds,
                 c='k', lw=2.5, zorder=3,
                 label='random audiogram')

        # plot individual trials
        for ix, (x, y, resp, color) in enumerate(zip(tested_freqs, tested_levels, responses, colors)):
            if ix < mapping['Bayesian'][1]:
                zorder=0
            elif ix < mapping['Cluster'][1]:
                zorder=2
            else:
                zorder=1

            plt.scatter(x, y, marker=marker_map[resp], c=color, s=size_map[resp],
                        edgecolors='white', lw=.5, alpha=1, zorder=zorder)

        # aesthetics
        plt.xscale('log')
        plt.xticks([125, 1000, 8000], [125, 1000, 8000])
        plt.minorticks_off()
        ax = plt.gca()
        ax.invert_yaxis()

        # Legend
        legend_elements = []
        paradigm_list = ['Randomized', '\nContinuous (R)', '\nCluster (R)']
        colors = ['#1E2023' if paradigm == 'Bayesian' else get_color(paradigm) for paradigm in ['Bayesian', 'Continuous', 'Cluster']]

        for paradigm, color in zip(paradigm_list, colors):
            legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label=paradigm))
            legend_elements.append(Line2D([0], [0], marker=u'$\u2713$', color=color, label='detected tone',
                                          markerfacecolor=color, markersize=18, linestyle='None', lw=1))
            legend_elements.append(Line2D([0], [0], marker='X', color=color, label='missed tone',
                                          markerfacecolor=color, markeredgecolor='white', lw=1,
                                          markersize=15, linestyle='None'))
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='                         '))
        legend_elements.append(Line2D([0], [0], color='k', lw=2.5, label='random\naudiogram'))

        plt.legend(handles=legend_elements, handlelength=1, handleheight=1,
                   labelspacing=0.5, loc='center left',  bbox_to_anchor=(1, 0.5),
                   ncol=1)

        plt.ylim([22, -17])
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Level (in dB HL)')
        plt.title('Global random audiogram')

        plt.tight_layout()

        return fig

    def plot_example_distances(self, plot_init=True, ylims=[]):
        """
        Plots the audiogram data for a specific participant along with trials for a specific paradigm and pred condition.

        Args:
            plot_init (bool): plots the data from the initialization phase if True (defaults to True)
            ylims (list): if plotting initialization data and you wish to plot tones limited to a specific dB range


        Returns:
            fig (matplotlib.figure.Figure): The output figure.

        """

        fig = plt.figure(figsize=self.FIG_SIZE)

        # Plot the global random audiogram
        audiogram_thresholds = self.random_audiogram.random_thresholds[0]
        audiogram_freqs = self.random_audiogram.frequencies[0]
        plt.plot(audiogram_freqs, audiogram_thresholds, c='k', lw=2.5, zorder=5, label='random\naudiogram')

        # Create the color and marker maps
        color_map = ['green' if resp == 1 else 'red' for resp in self.trials_data.responses]
        marker_map = {1: u'$\u2713$', 0: 'X'}
        size_map = {1: 400, 0: 100}

        print(self.trials_data.len_init.iloc[0])

        # Plot the trials
        for idx, (freq, level, resp, random_threshold) in enumerate(zip(self.trials_data.tested_frequencies,
                                                                        self.trials_data.tested_levels,
                                                                        self.trials_data.responses,
                                                                        self.trials_data.random_threshold)):

            if idx < self.trials_data.len_init.iloc[0] and not plot_init:
                continue

            if plot_init and len(ylims) == 2:
                if level < ylims[0] or level > ylims[1]:
                    continue

            plt.scatter(freq, level,
                        c=color_map[idx], marker=marker_map[resp], s=size_map[resp],
                        edgecolors='white', lw=.5, alpha=1)
            plt.vlines(x=freq, ymin=level, ymax=random_threshold,
                       # color=color_map[idx], linestyles=':',
                       color='.7', lw=2,
                       alpha=0.7, zorder=0)

        legend_elements = []
        # legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='Tested tones'))
        legend_elements.append(Line2D([0], [0], marker=u'$\u2713$', color='green', label='detected tone',
                                      markerfacecolor='green', markersize=15, linestyle='None', lw=.5))
        legend_elements.append(Line2D([0], [0], marker='X', color='red', label='missed tone',
                                      markerfacecolor='red', markeredgecolor='white', lw=.5,
                                      markersize=11, linestyle='None'))
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='              '))
        legend_elements.append(Line2D([0], [0], color='k', lw=2.5, label='random\naudiogram'))
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='                         '))
        legend_elements.append(Line2D([0, 0], [0, 1], color='.7', alpha=.7, lw=2, label='distance from\nthreshold'))

        # Aesthetics
        plt.xscale('log')
        plt.xticks([125, 1000, 8000], [125, 1000, 8000])
        # plt.ylim([42, -16])
        plt.minorticks_off()
        plt.gca().invert_yaxis()
        plt.legend(handles=legend_elements, loc='center left',  bbox_to_anchor=(1, 0.5))
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Level (in dB HL)')
        plt.title(f"Tone distance from threshold")  # (example data from {self.paradigm}{'' if self.paradigm in ['3AFC', 'Bayesian'] else f'/{self.pred}'})")

        plt.tight_layout()

        return fig

    def plot_example_sigmoid(self):

        ylims = [0, 1]

        color_map = ['green' if resp == 1 else 'red' for resp in self.trials_data.responses]
        # markers
        marker_map = [u'$\u2713$' if resp == 1 else 'X' for resp in self.trials_data.responses]
        size_map = self.trials_data['responses'].replace({1: 350, 0: 100})

        fig = plt.figure(figsize=self.FIG_SIZE)

        plt.vlines(0, *ylims, colors='.7', linestyles=':', zorder=1)
        # plt.vlines(0, 0.48, 0.52, colors='k', zorder=1.5, lw=2)
        plt.vlines(self.sigmoid_data.distance_p50.unique()[0], *ylims, colors='.7', lw=2, zorder=1)
        plt.hlines(.5, self.sigmoid_data.random_distance.min(), self.sigmoid_data.random_distance.max(),
                   colors='.7', linestyles=':', zorder=1)


        plt.plot(self.sigmoid_data.random_distance, self.sigmoid_data.sigmoid_probas, c='k', zorder=3)
        plt.scatter(self.sigmoid_data.distance_p50.unique()[0], .5, s=50, c='k', zorder=4, label='p50')

        # Plot each point individually with its own marker
        for i, (dist, resp, m, s, c) in enumerate(zip(self.trials_data.random_distance,
                                                self.trials_data.responses,
                                                marker_map,
                                                size_map,
                                                color_map)):
            plt.scatter(dist, resp, marker=m, s=s, facecolors=c, edgecolors='white', lw=.75, alpha=1, zorder=5)

        ax = plt.gca()
        ax.annotate('', xy=(self.sigmoid_data.distance_p50.unique()[0], 0.5), xytext=(0, 0.5),
                           arrowprops=dict(facecolor='k', edgecolor='k', arrowstyle='->', lw=3, zorder=10))


        plt.yticks([0, .5, 1], ["0", "0.5", "1"])
        plt.xticks([-15, -10, -5, 0, 5, 10, 15, 20])
        xlim_min = 2 * math.floor(self.trials_data.random_distance.min() / 2)
        xlim_max = 5 * math.ceil(self.trials_data.random_distance.max() / 5)
        # plt.xlim([xlim_min, -xlim_min])
        plt.xlim([-16, 21])
        plt.ylim(ylims[0] - .05, ylims[1] + .05)

        legend_elements = []
        # legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='Tested tones'))
        legend_elements.append(Line2D([0], [0], marker=u'$\u2713$', color='green', label='detected tone',
                                      markerfacecolor='green', markersize=15, linestyle='None', lw=.5))
        legend_elements.append(Line2D([0], [0], marker='X', color='red', label='missed tone',
                                      markerfacecolor='red', markeredgecolor='white', lw=.5,
                                      markersize=11, linestyle='None'))
        legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='none', label='              '))
        legend_elements.append(Line2D([0], [0], color='k', marker='o', markersize=7, linestyle='None', label='p50'))


        plt.legend(handles=legend_elements, loc='center left',  bbox_to_anchor=(1, 0.5))

        plt.title(f"p50")  # (example data from {self.paradigm}{'' if self.paradigm in ['3AFC', 'Bayesian'] else f'/{self.pred}'})")
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
                         facecolors='r' if anomaly_type == 'p50 outside range' else '.7', alpha=.1, zorder=0)
        ax.hlines(.5, min(xsigmoid), max(xsigmoid),
                  colors='.7', zorder=1)
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
                #             edgecolors='.7', facecolors='None', alpha=.1, s=2,
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
    def plot_paradigm_comparison(ax, paradigms_data, paradigms_posthoc, var, show_ns=False):
        ax.hlines(0, -0.5, 3.5, '.7')

        # Define positions
        pos_dict = {"Bayesian": 0, "Continuous": 1, "Cluster": 2, "3AFC": 3}

        for n, paradigm in enumerate(pos_dict.keys()):
            mean_val = paradigms_data.loc[paradigms_data.paradigm == paradigm, var].mean()
            sem_val = paradigms_data.loc[paradigms_data.paradigm == paradigm, var].sem()

            ax.bar(n, mean_val, width=.5, color=get_color(paradigm, 'none' if paradigm in ['Cluster', 'Continuous'] else None))
            ax.errorbar(n, mean_val, sem_val, color='black', elinewidth=3)

        # Plot significance
        for _, row in paradigms_posthoc.iterrows():
            A, B, p_corr = row['A'], row['B'], row['p-corr']

            # Skip non-significant results unless show_ns is True
            if p_corr > 0.05 and not show_ns:
                continue

            y_max = min(paradigms_data.loc[paradigms_data.paradigm == A, var].mean(),
                        paradigms_data.loc[paradigms_data.paradigm == B, var].mean())

            # y_pos = y_max - 3.2 + pos_dict[B]
            y_pos = (y_max + min([pos_dict[A], pos_dict[B]])) - 3.5

            # Draw a horizontal line between A and B
            ax.hlines(y_pos, pos_dict[A], pos_dict[B], 'black')
            ax.vlines([pos_dict[A], pos_dict[B]],
                       y_pos + .25, y_pos, 'black')

            # Draw an asterisk or "n.s." above the line
            if p_corr < 0.05:
                ax.text((pos_dict[A] + pos_dict[B]) / 2, y_pos - .1,
                         '*', ha='center', va='center')
            else:
                ax.text((pos_dict[A] + pos_dict[B]) / 2, y_pos - .5,
                         'n.s.', ha='center', va='top', fontsize=10)

        ax.set_ylim([2, -12 if var == 'distance_p50' else -8])
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylabel('p50 threshold distance' if var == 'distance_p50' else 'Mean threshold')
        ax.set_xticks([-0.25, 0.75, 1.75, 2.75])
        ax.set_xticklabels(['Random', 'Continuous', 'Cluster', '3AFC'], rotation=25)
        ax.tick_params(axis='x', which='both', length=0)


    @staticmethod
    def plot_pred_comparison(ax, continuous_data, cluster_data,
                             continuous_posthoc, cluster_posthoc,
                             var, show_ns=False):

        ax.hlines(0, -1, 9, '.7')

        # Define positions
        pred_dict = {'R': 0, 'T': 1, 'F': 2, 'FT': 3}

        for n, (paradigm, paradigm_data, comparisons) in enumerate(zip(['Continuous', 'Cluster'],
                                                                       [continuous_data, cluster_data],
                                                                       [continuous_posthoc, cluster_posthoc])):

            for pred, m in pred_dict.items():
                ax.bar(m + 5 * n,
                        paradigm_data.loc[(paradigm_data.pred == pred), var].mean(),
                        color=pred_palette(pred))
                ax.errorbar(m + 5 * n, paradigm_data.loc[(paradigm_data.pred == pred), var].mean(),
                             paradigm_data.loc[paradigm_data.pred == pred, var].sem(),
                             color='black', elinewidth=3)

            # Plot significance
            for _, row in comparisons.iterrows():
                A, B, p_corr = row['A'], row['B'], row['p-corr']

                # Skip non-significant results unless show_ns is True
                if p_corr > 0.05 and not show_ns:
                    continue

                # Calculate y position for the line and asterisk/n.s.
                y_pos = -.7 * (pred_dict[A] + pred_dict[B]) - 2.5

                # Draw a horizontal line between A and B
                ax.hlines(y_pos, pred_dict[A] + 5 * n, pred_dict[B] + 5 * n, 'black')
                ax.vlines([pred_dict[A] + 5 * n, pred_dict[B] + 5 * n],
                           y_pos + .25, y_pos, 'black')

                # Draw an asterisk or "n.s." above the line
                if p_corr < 0.05:
                    ax.text((pred_dict[A] + pred_dict[B]) / 2 + 5 * n, y_pos - .03,
                             '*', ha='center', va='bottom')
                else:
                    ax.text((pred_dict[A] + pred_dict[B]) / 2 + 5 * n, y_pos - .03,
                             'n.s.', ha='center', va='bottom', fontsize=10)

        ax.set_ylim([1, -6])
        ax.set_xlim([-1, 9])
        ax.set_xticks([1.5, 6.5])
        ax.set_xticklabels(['Continuous', 'Cluster'])

    def p50_barplot(self, var, paradigms_data, continuous_data, cluster_data,
                    paradigms_posthoc, continuous_posthoc, cluster_posthoc, show_ns=False):

        fig = plt.figure(figsize=(8, 5.33))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Break down your plotting to these two main components
        self.plot_paradigm_comparison(ax1, paradigms_data, paradigms_posthoc, var, show_ns)
        self.plot_pred_comparison(ax2, continuous_data, cluster_data, continuous_posthoc, cluster_posthoc, var, show_ns)

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


    @staticmethod
    def plot_bar(mean, error, color, x):
        plt.bar(x, mean, color=color)
        plt.errorbar(x, mean, error,  color='black', elinewidth=3)

    @staticmethod
    def get_mean_FAR_and_error(catch_trials_data):
        mean = catch_trials_data.false_alarm_rate.mean()
        error = catch_trials_data.false_alarm_rate.sem()
        return mean, error

    def catch_trials_barplot(self, catch_trials_data):

        update_plot_params()

        fig = plt.figure(figsize=[7, 5.33])

        # Bayesian
        mean, error = self.get_mean_FAR_and_error(catch_trials_data.loc[catch_trials_data.paradigm == 'Bayesian'])
        self.plot_bar(mean, error, paradigms_palette('Bayesian'), -2)

        # Cluster and Continuous
        for n, (paradigm, paradigm_data) in enumerate(zip(['Continuous', 'Cluster'],
                                                          [catch_trials_data.loc[catch_trials_data.paradigm == 'Continuous'],
                                                           catch_trials_data.loc[catch_trials_data.paradigm == 'Cluster']])):

            for m, pred in enumerate(['none', 'time', 'frequency', 'both']):
                mean, error = self.get_mean_FAR_and_error(paradigm_data.loc[paradigm_data.pred == pred])
                self.plot_bar(mean, error, pred_palette(pred), 5 * n + m)

        plt.ylim([0, .4])
        plt.xlim([-3.5, 9])
        plt.yticks([0, .1, .2, .3, .4], ['.0', '.1', '.2', '.3', '.4'])
        plt.xticks([-2, 1.5, 6.5], ['Randomized', 'Continuous', 'Cluster'])
        plt.ylabel('False alarm rate')
        plt.tight_layout()

        return fig

    def plot_linear_regression(self, data, xvar, yvar, r_values, labels=None):

        if labels == None:
            labels = xvar, yvar

        if len(data.pred.unique()) == 4:
            preds = ['none', 'time', 'frequency', 'both']
            x_inset = [0, 1, 2, 3]
        else:
            preds = ['time', 'frequency', 'both']
            x_inset = [0, 1, 2]


        update_plot_params()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[10, 5.33], sharex='all')

        for idx, paradigm in enumerate(['Continuous', 'Cluster']):

            mainax = plt.sca(axs[idx])

            for pred_idx, pred in enumerate(preds):

                pred_data = data.loc[(data.paradigm == paradigm) & (data.pred == pred)]

                x = pred_data[xvar].to_numpy()
                y = pred_data[yvar].to_numpy()

                a, b = np.polyfit(x, y, 1)
                plt.plot(x, a * x + b, lw=1.5, color=pred_palette(pred), zorder=5)  # alpha=.95,
                plt.scatter(x, y, s=30, facecolors=pred_palette(pred), edgecolors='None', alpha=.65)

            plt.xticks()
            plt.yticks()
            # plt.legend(fontsize='small')
            plt.ylabel(labels[1])
            plt.xlabel(labels[0])
            plt.ylim([8, -8])
            plt.ylim([data[yvar].max() + 1 if data[yvar].max() > 6 else 7, data[yvar].min() - 1])  # , p50_Clus_Cont.max() + 1])
            # plt.xlim([-0.02, 0.72])
            plt.title(f'{paradigm}')


            # ADD INSET AX
            insetax = plt.gca().inset_axes([0.7, 0.115, .25, .25])
            # plot R bars
            insetax.bar(x_inset,
                        [r_values.loc[(r_values.paradigm == paradigm) & (r_values.pred == p), 'R2'].iloc[0] for p in preds],
                        color=[pred_palette(p) for p in preds])

            # plot significance stars
            for i, p in enumerate(preds):
                r2_val = r_values.loc[(r_values.paradigm == paradigm) & (r_values.pred == p), 'R2'].iloc[0]
                p_val = r_values.loc[(r_values.paradigm == paradigm) & (r_values.pred == p), 'p_value'].iloc[0]
                if p_val <= 0.05:
                    insetax.scatter(i, r2_val + 0.15, marker="$*$", c='k', s=100)

            insetax.set_xticks(x_inset, [translate_conditions(pred) for pred in preds])
            insetax.set_yticks([0, .5, 1], [0, 0.5, 1])
            insetax.set_title("R$^2$")
            # weight='bold')
            insetax.patch.set_alpha(.2)

        plt.tight_layout()

        return fig


class ClusterPlotter:

    def __init__(self):
        pass

    @staticmethod
    def custom_legend(fig, paradigms, n_pcs):

        preds = ['none', 'time', 'frequency', 'both']
        pred_labels = [translate_conditions(pred) + "   " for pred in preds]

        # Creating the legend manually for paradigms
        patches = [mpatches.Patch(facecolor=paradigms_palette(paradigm), alpha=0.4) for paradigm in paradigms]
        markers = [Line2D([0], [0], marker='o', color=paradigms_palette(paradigm), linestyle='None') for paradigm in paradigms]
        legend_elements = [(marker, patch) for marker, patch in zip(markers, patches)]

        # Creating the legend manually for preds
        pred_markers = [Line2D([0], [0], marker='o', color=pred_palette(pred), linestyle='None') for pred in preds]
        legend_elements.extend(pred_markers)

        # Add the legend to the figure
        fig.legend(legend_elements, list(paradigms) + pred_labels,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                   loc='lower left', bbox_to_anchor=(1/(n_pcs+2), 1/(n_pcs-2)), ncol=2)
                   # columnspacing=0.5)

        # for x, y in zip([0, .195, .39, .59], [.655, .463, .265, .07]):
        #     fig.legend(legend_elements, list(paradigms) + pred_labels,
        #                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        #                loc='lower left', bbox_to_anchor=(x, y), ncol=2, columnspacing=0.5)

    def plot_pcs(self, table, pca, pcs, n_pcs):

        update_plot_params()

        # Create the subplots
        fig, axs = plt.subplots(n_pcs-1, n_pcs-1, figsize=(3 * (n_pcs-1), 3 * (n_pcs-1)))

        # Create a new column for paradigm and pred
        table['paradigm'] = table.index.str.split(',').str[0]
        table['pred'] = table.index.str.split(',').str[-1]

        paradigms = table['paradigm'].unique()

        self.custom_legend(fig, paradigms, n_pcs)

        # Get all combinations of PCs
        combs = list(combinations(range(n_pcs), 2))

        # Calculate the number of subplots needed
        n_subplots = len(combs)

        for subplot_idx in range(n_subplots):
            pc1, pc2 = combs[subplot_idx]

            ax = axs[pc1, pc2 - 1]
            var_exp = np.round((pca.explained_variance_ratio_[pc1] + pca.explained_variance_ratio_[pc2]) * 100, 1)

            for paradigm in paradigms:

                # Subset the data
                paradigm_data = pcs[table['paradigm'] == paradigm]
                paradigm_data_subset = paradigm_data[:, [pc1, pc2]]

                unique_preds = table.loc[table['paradigm'] == paradigm, 'pred'].unique()

                # Differentiate color by paradigm and pred
                if paradigm in ['Randomized', '3AFC']:
                    color = paradigms_palette(paradigm)
                elif paradigm in ['Continuous', 'Cluster']:
                    color = [pred_palette(pred.strip()) for pred in unique_preds]

                ax.scatter(paradigm_data_subset[:, 0], paradigm_data_subset[:, 1],
                           facecolors=color, edgecolors='w', lw=.75,
                           s=75, label=None)

                # Check if the number of unique points is >= 3 for the convex hull to make sense
                if len(np.unique(paradigm_data_subset, axis=0)) >= 3:
                    # Compute the convex hull for the current paradigm
                    hull = ConvexHull(paradigm_data_subset)

                    # Get the hull points
                    hull_points = paradigm_data_subset[hull.vertices]

                    # Fill the hull with a color
                    ax.fill(hull_points[:, 0], hull_points[:, 1], facecolor=paradigms_palette(paradigm), alpha=var_exp/100, zorder=0)

                elif len(np.unique(paradigm_data_subset, axis=0)) == 1:
                    ax.scatter(paradigm_data_subset[:, 0], paradigm_data_subset[:, 1],
                               facecolors=paradigms_palette(paradigm), edgecolors=None, lw=0,
                               s=400, alpha=var_exp/100, zorder=0)

            ax.set_xlabel(f'PC{pc1+1}')
            ax.set_ylabel(f'PC{pc2+1}')
            ax.set_title(f'PC{pc1+1} vs PC{pc2+1} ({var_exp}%)')

        # Remove unused subplots in the lower left triangle
        for i in range(n_pcs - 1):
            for j in range(i):
                axs[i, j].axis('off')

        plt.tight_layout()

        # remove the added columns no longer useful
        table = table.drop(['paradigm', 'pred'], axis=1)

        return fig

    @staticmethod
    def plot_skeleton_find_lines(dendrogram):

        xpairs, ypairs = [], []
        for xcoords, ycoords in zip(dendrogram['dcoord'], dendrogram['icoord']):

            # Find x coordinates
            xsublist_pairs = [(xcoords[i], xcoords[i + 1]) for i in range(len(xcoords) - 1)]
            xpairs += xsublist_pairs

            # Find y coordinates
            ysublist_pairs = [(ycoords[i], ycoords[i + 1]) for i in range(len(ycoords) - 1)]
            ypairs += ysublist_pairs

        # Create DataFrame with coordinates for all lines in the dendrogram
        line_coords = pd.DataFrame(np.concatenate([np.array(xpairs), np.array(ypairs)], axis=1), columns=['x1', 'x2', 'y1', 'y2'])

        # Plot lines of the skeleton and find lines to color
        lines_to_color = pd.DataFrame()
        for idx, line in line_coords.iterrows():
            xcoords = line[:2]
            ycoords = line[2:]

            if any(line[:2] == 0):
                lines_to_color = pd.concat([lines_to_color, pd.DataFrame(line).T])
            else:
                plt.plot(xcoords, ycoords, '#B4B8BF', lw=2)

        return lines_to_color

    def plot_dendrogram(self, clusters, dendrogram, table):

        update_plot_params()

        fig = plt.figure(figsize=(4.5, 5.33))

        # Plot skeleton and find lines corresponding to the labels
        lines = self.plot_skeleton_find_lines(dendrogram)
        lines.sort_values(by='y1', inplace=True)

        # Find the labels
        labels = [table.index[idx] for idx in dendrogram['leaves']]
        for (_, line), label in zip(lines.iterrows(), labels):

            paradigm = label.split(', ')[0]
            pred = label.split(', ')[-1]

            color = paradigms_palette(paradigm) if paradigm in ['3AFC', 'Randomized'] else pred_palette(pred)

            plt.plot(line[:2], line[2:], color=color, lw=5)

        labels = [f"{' ':<3}{label.split(', ')[0]}" if label.split(', ')[0] in ['3AFC', 'Randomized'] else f"   {translate_conditions(label.split(', ')[1]):<4} |   " + label.split(', ')[0] for label in labels]

        ax = plt.gca()
        plt.xlim([1.5, 0])
        plt.xlabel('Cosine distance')
        plt.yticks(lines['y1'], labels)
        ax.yaxis.tick_right()
        plt.tick_params(axis='y', length=0) # set y-tick length to 0

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        return fig


class SupplPlots:

    @staticmethod
    def lmm_assumptions(data):
        """
        Check assumptions.

        Plots: Linearity, normality and homoscedasticity checks

        """

        import seaborn as sns

        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # Linearity: Plotting predicted values vs residuals
        sns.regplot(x='predicted', y='residuals', data=data, ax=axs[0], scatter_kws={'alpha':0.5})
        axs[0].set_title('Residuals vs. Predicted values')

        # Normality: Histogram of residuals
        sns.histplot(data['residuals'], kde=True, ax=axs[1])
        axs[1].set_title('Histogram of residuals')

        # Homoscedasticity: Scale-Location plot
        sns.regplot(x='predicted', y=abs(data['residuals']), data=data, scatter_kws={'alpha': 0.5}, ax=axs[2])
        axs[2].set_title('Scale-Location')

        # Show the plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def example_staircase(data_3AFC, threshold, reversal_indices, reversal_levels, chosen_freq):

        # Find the index of the first tone
        # min_idx = data_3AFC.n_tone.min()

        # Plot a horizontal line for the corresponding threshold
        plt.hlines(y=threshold,
                   xmin=0, xmax=data_3AFC.n_tone.max(),
                   colors='.7', lw=2, zorder=1)

        # Plot the line of tested level ~ freq
        plt.plot(data_3AFC.n_tone[:-1], data_3AFC.tested_levels[:-1], 'k:', lw=1, zorder=0)

        # Show the reversals used for threshold computation
        plt.scatter(reversal_indices, reversal_levels, marker='o', s=225, edgecolors='.7', facecolors='white', lw=2, zorder=3)

        marker_map = {1: u'$\u2713$', 0: 'X'}

        # Loop over the rows in the DataFrame
        for index, row in data_3AFC.iterrows():
            if index == len(data_3AFC) - 1:
                continue
            positive_detection = row.responses
            if positive_detection:
                plt.scatter(row['n_tone'], row['tested_levels'],
                         marker=marker_map[row.responses], s=150, facecolors='g', zorder=4)  # , edgecolors='white', lw=.0001)
            else:
                plt.scatter(row['n_tone'], row['tested_levels'],
                         marker=marker_map[row.responses], s=75, facecolors='r', edgecolors='white', lw=.01, zorder=4)

        # Add text in the top-right corner with the selected frequency
        plt.annotate(f'{chosen_freq} Hz', xy=(1, 1), xytext=(-10, -10),
                     xycoords='axes fraction', textcoords='offset points',
                     horizontalalignment='right', verticalalignment='top',
                     fontweight='bold')

        # Add labels to the axes
        plt.xlabel('Tone number')
        plt.ylabel('Tested level')
        plt.title('3-AFC staircase example data')

    @staticmethod
    def example_3afc_audiogram(audiogram_3AFC, participant):

        plt.hlines([-20, -10, 0, 10, 20], 0, 12000, colors='.7', alpha=0.5)
        plt.plot(audiogram_3AFC.sort_values('frequencies').frequencies,
                 audiogram_3AFC.sort_values('frequencies').thresholds,
                 label=participant, color='red')
        plt.scatter(audiogram_3AFC.frequencies, audiogram_3AFC.thresholds,
                    label=None, edgecolors='red', facecolors="white", zorder=3)
        plt.xscale('log')
        plt.ylim([30, -30])
        plt.xticks([125, 250, 500, 1000, 2000, 4000, 8000], [125, 250, 500, 1000, 2000, 4000, 8000])
        plt.xlim([100,10000])
        plt.ylabel("Threshold (dB HL)")
        plt.xlabel('Frequency (Hz)')
        plt.title('3-AFC example individual results')
