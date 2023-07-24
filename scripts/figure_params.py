import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import get_path


def update_plot_params():
    plt.rcParams.update(
        {'font.size': 16,
         'font.family': 'Helvetica Neue',
         'axes.titleweight': 'bold',
         # 'axes.labelcolor': '#423E3E',
         'axes.titlesize': 16,
         'axes.labelsize': 15,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14,
         'legend.fontsize': 15,
         'savefig.directory': get_path("figures")}
    )


def print_rc_params():
    import matplotlib.font_manager

    fonts = []
    for font in matplotlib.font_manager.fontManager.ttflist:
        # print(font.weight)
        fonts.append(font.name)
        print(font.name)


def fig_params():
    params = pd.Series({'page_width': 16 / 2.54,
                        'title_font': 'Arial',
                        '2C_dims': [9, 5],
                        })

    return params


def paradigms_palette(paradigm):
    palette = {'Bayesian': '#4A5666',
               'Cluster': '#bb8694',
               'Continuous': '#5E586A',
               '3AFC': '#ED5153'
               }
    return palette[paradigm]


def pred_palette(pred):
    palette = {'none': '#292F36',
               'time': '#52BFB6',
               'frequency': '#E3BC59',
               'both': '#E03639'
               }
    return palette[pred]


def get_color(paradigm, pred=None):
    if pred and pred in ['both', 'frequency', 'time', 'none']:
        return pred_palette(pred)
    else:
        return paradigms_palette(paradigm)
