from scipy.stats import entropy
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import re
import plot_audio
from collections import OrderedDict
import math
from scipy import stats


# Some Helper functions we need
def normalize_mean_sub(data):
    return np.array(data) - np.mean(data)


def scipy_entropy(data):
    return entropy(data, base=2)


def probability_hist(data, bins):
    dist = np.histogram(data, bins=bins)
    return dist[0] / np.sum(dist[0])


def get_line_style(count):
    # Define line styles
    line_styles = OrderedDict(
        [('solid', (0, ())),
         ('dashdot', '-.'),
         ('loosely dotted', (0, (1, 10))),
         ('dotted', (0, (1, 5))),
         ('densely dotted', (0, (1, 1))),

         ('loosely dashed', (0, (5, 10))),
         ('dashed', (0, (5, 5))),
         ('densely dashed', (0, (5, 1))),
         ('_densely dashed', (0, (3, 1))),

         ('loosely dashdotted', (0, (3, 10, 1, 10))),
         ('dashdotted', (0, (3, 5, 1, 5))),
         ('densely dashdotted', (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

    used_styles = ['solid', 'densely dotted', '_densely dashed', 'densely dashdotdotted', 'densely dashdotted']

    return line_styles.get(used_styles[count])


def find_div(y_max):
    div = None

    if 0 < y_max < 1:
        div = 1
    elif 1 < y_max < 10:
        div = 1
    elif 10 < y_max < 100:
        div = 10
    elif 100 < y_max < 1000:
        div = 100
    elif 1000 < y_max < 10000:
        div = 1000
    elif 10000 < y_max < 100000:
        div = 10000
    elif 100000 < y_max < 1000000:
        div = 100000

    return div


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def get_experiment_labels():
    return ['PA', 'AA', 'PA + H', 'AA + H']


def get_displayed_name(experiment):
    names = get_experiment_labels()
    if experiment.lower().__contains__('base'):
        return names[0]
    elif re.search('^.*entro.*attacker$', experiment):
        return names[3]
    elif re.search('^.*attacker$', experiment):
        return names[1]
    elif re.search('^.*entropy$', experiment):
        return names[2]


def get_entropy_labels_and_index(labels):
    # Get labels to be used in plots
    actual_order = get_experiment_labels()
    label_order = []
    label_index = []

    # Iterate over labels
    for label in labels:
        label_name = get_displayed_name(label)
        label_order.append(label_name)
        label_index.append(actual_order.index(label_name))

    return label_order, np.array(label_index)


def get_light_display_name(label):
    names = ['Light_4bulbs', 'Light_4bulbs2', 'Light_4bulbs_no_outlier']
    if label == names[0]:
        return '4 Bulbs 1'
    if label == names[1]:
        return '4 Bulbs 2'
    if label == names[2]:
        return '4 Bulbs 1'


def get_light_followup_names(labels):
    new_labels = []
    for label in labels:
        new_labels.append(get_light_display_name(label))

    return new_labels, np.arange(len(new_labels))


def get_gas_displayed_names(label):
    names = ['Humid_close', 'Humid_close_no_outlier', 'Humid_1m', 'Humid_1m_no_outlier']
    if label == names[0]:
        return '1cm A'
    if label == names[1]:
        return '1cm A'
    if label == names[2]:
        return '1m A'
    if label == names[3]:
        return '1m A'


def get_gas_followup_names(labels):
    new_labels = []
    for label in labels:
        new_labels.append(get_gas_displayed_names(label))

    return new_labels, np.arange(len(new_labels))


def draw_experiment_entropy_bar_plot(data, save_path, caption, bin, modality, single_bar=True, exclude_list=[],
                                     arrange_order=False, is_gas_followup=False, is_light_followup=False,
                                     is_audio_followup=False):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    # Store data for plotting
    colocated_means = []
    colocated_error = []
    non_colocated_means = []
    non_colocated_error = []

    labels = []  # [Experiment_name1, Experiment_name2]

    # Get all relevant information from the dict
    for experiment in data[modality]:
        if exclude_list.__contains__(experiment):  # skip excluded experiments
            continue

        co_list = []
        non_co_list = []
        labels.append(experiment)

        for entro, name in data[modality][experiment][bin]:
            if name.__contains__('2'):
                non_co_list.append(entro)
            else:
                co_list.append(entro)

        colocated_means.append(np.mean(co_list))
        colocated_error.append(np.std(co_list))
        non_colocated_means.append(np.mean(non_co_list))
        non_colocated_error.append(np.std(non_co_list))

    # If we consider all experiments in the follow-up, let's rearrange bars a bit
    followup_flag = False
    if is_audio_followup:
        if sorted(labels) == sorted(plot_audio.get_followup_names()):
            labels = plot_audio.get_followup_names()
            followup_flag = True

    # Get label order right for different modalities
    if arrange_order:
        labels, x = get_entropy_labels_and_index(labels)
    elif is_gas_followup:
        labels, x = get_gas_followup_names(labels)
    elif is_light_followup:
        labels, x = get_light_followup_names(labels)
    elif is_audio_followup:
        labels, x = plot_audio.get_followup_labels(labels)
    else:
        x = np.arange(len(labels))  # the label locations

    # Bar width
    width = 0.55

    # Set figure size
    plt.figure(figsize=(6.5, 4))

    colors = ['#c14a09', '#06b48b', '#9d5783', '#287c37']

    # Choose to plot only colocated devices or both colocated and non-colocated
    if single_bar:
        # This is what we want: the color is called 'pea green': #8eab12 from https://xkcd.com/color/rgb/
        # NEW color: 'darkish green': #287c37
        p = plt.bar(x, colocated_means, width=width, color=colors[2], yerr=colocated_error, label='Coloc.',
                    zorder=3, error_kw=dict(capsize=6, capthick=3, elinewidth=3), linewidth=0.6)
    else:
        # This is not what we want since we don't want to see the non-colocated entropy
        plt.bar(x - width / 2, colocated_means, width, yerr=colocated_error, label='Colocated', zorder=3,
                error_kw=dict(capsize=4, capthick=2))
        plt.bar(x + width / 2, non_colocated_means, width, yerr=non_colocated_error, label='Non-colocated',
                zorder=3, error_kw=dict(capsize=4, capthick=2))

    # Add a grid to the plot
    plt.grid(True, axis='y', zorder=0)

    # Set range for X and Y axes
    plt.xticks(x, labels)

    # Set y_max and step
    _, _, _, y2 = plt.axis()
    div = find_div(y2)
    y_max = round(y2 / div, 1)
    # plt.yticks(np.arange(0, y_max, step=step))
    plt.yticks(np.arange(0, 1.2, step=0.25))

    # Plot stuff for paper or analysis
    if followup_flag:
        plt.tick_params(axis='both', labelsize=12)
        plt.xticks(rotation=30)
    else:
        plt.tick_params(axis='both', labelsize=22)

    # Set font sizes of axes and their ticks
    plt.xlabel('Experiment', fontsize=26)
    plt.ylabel('Entropy', fontsize=26)

    # Have ticks on left and right side of Y-axis
    plt.gca().yaxis.set_ticks_position('both')

    # Ticks inside the plot look nicer IMHO
    plt.tick_params(axis='y', direction='in')

    # Add annotations to the bars
    plt.bar_label(p, fmt='%.2f', label_type='center', fontsize=20, color='white', weight='bold')

    plt.legend(loc='best', ncol=1, borderaxespad=0., fontsize=21,
               handlelength=1, handletextpad=0.2, columnspacing=0.6)

    # Save plots (let's do both PDF and SVG)
    to_save = os.path.join(save_path, f'{caption}_{bin}')
    plt.savefig(to_save + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(to_save + '.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.close('all')


def draw_experiment_distance_bar_plot(data, save_path, caption, normalize=False, exclude_list=[], arrange_order=False,
                                      is_gas_followup=False, is_light_followup=False):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    # We save relevant data for plotting as well as use to get avg values
    labels = []
    avg_distances_colocated = {}
    avg_distances_non_colocated = {}
    colocated_error = {}
    non_colocated_error = {}
    colocated_per_time_interval = {}
    non_colocated_per_time_interval = {}

    # gather all needed data and save them to pass to matplotlib
    for experiment in data:
        if exclude_list.__contains__(experiment):
            continue

        for time_split in data[experiment]:
            time_split_interval = {}

            for modality in data[experiment][time_split]:
                if not time_split_interval.__contains__(modality):
                    time_split_interval[modality] = {}
                    time_split_interval[modality]['colocated'] = []
                    time_split_interval[modality]['non-colocated'] = []

                if not avg_distances_colocated.__contains__(modality):
                    avg_distances_colocated[modality] = []
                    avg_distances_non_colocated[modality] = []
                    colocated_error[modality] = []
                    non_colocated_error[modality] = []
                    colocated_per_time_interval[modality] = []
                    non_colocated_per_time_interval[modality] = []

                for name1 in data[experiment][time_split][modality]:
                    for name2 in data[experiment][time_split][modality][name1]:
                        distance = data[experiment][time_split][modality][name1][name2]
                        if name1.__contains__('2') or name2.__contains__('2'):
                            time_split_interval[modality]['non-colocated'].append(distance)
                        else:
                            time_split_interval[modality]['colocated'].append(distance)

                if time_split == 'full':
                    avg_distances_colocated[modality].append(np.mean(time_split_interval[modality]['colocated']))
                    colocated_error[modality].append(np.std(time_split_interval[modality]['colocated']))
                    avg_distances_non_colocated[modality].append(
                        np.mean(time_split_interval[modality]['non-colocated']))
                    non_colocated_error[modality].append(np.std(time_split_interval[modality]['non-colocated']))
                else:
                    colocated_per_time_interval[modality].append(np.mean(time_split_interval[modality]['colocated']))
                    non_colocated_per_time_interval[modality].append(
                        np.mean(time_split_interval[modality]['non-colocated']))
                    time_split_interval[modality]['non-colocated'].clear()
                    time_split_interval[modality]['colocated'].clear()
        labels.append(experiment)

        for modality in avg_distances_colocated:
            if time_split == 'full':
                continue

            avg_distances_colocated[modality].append(np.mean(colocated_per_time_interval[modality]))
            colocated_error[modality].append(np.mean(colocated_per_time_interval[modality]))
            avg_distances_non_colocated[modality].append(np.mean(non_colocated_per_time_interval[modality]))
            non_colocated_error[modality].append(np.std(non_colocated_per_time_interval[modality]))
            colocated_per_time_interval[modality].clear()
            non_colocated_per_time_interval[modality].clear()

    if arrange_order:
        labels, x = get_entropy_labels_and_index(labels)
    elif is_gas_followup:
        labels, x = get_gas_followup_names(labels)
    elif is_light_followup:
        labels, x = get_light_followup_names(labels)
    else:
        x = np.arange(len(labels))  # the label locations

    for modality in avg_distances_colocated:
        if normalize:
            max_val_co = np.max(
                np.max(np.array(avg_distances_colocated[modality]) + np.array(colocated_error[modality])))
            max_val_non_co = np.max(
                np.array(avg_distances_non_colocated[modality]) + np.array(non_colocated_error[modality]))
            max_val = max(max_val_co, max_val_non_co)
            avg_distances_colocated[modality] = np.array(avg_distances_colocated[modality]) / max_val
            colocated_error[modality] = np.array(colocated_error[modality]) / max_val
            avg_distances_non_colocated[modality] = np.array(avg_distances_non_colocated[modality]) / max_val
            non_colocated_error[modality] = np.array(non_colocated_error[modality]) / max_val

        # Adjust hatch size and color
        plt.rcParams.update({'hatch.color': 'white'})
        plt.rcParams.update({'hatch.linewidth': 3})

        # Color bars nicely
        colors = ['#c14a09', '#06b48b', '#9d5783', '#287c37']

        # Set figure size
        plt.figure(figsize=(6.5, 4))

        # Width of the bars
        width = 0.365

        # Create bar plots
        plt.bar(x - width / 2, avg_distances_colocated[modality], width, yerr=colocated_error[modality],
                color=colors[1],
                hatch='/', label='Coloc.', zorder=3, error_kw=dict(capsize=6, capthick=3, elinewidth=3), linewidth=0.6)

        plt.bar(x + width / 2, avg_distances_non_colocated[modality], width, yerr=non_colocated_error[modality],
                color=colors[0],
                hatch='.', label='Non-coloc.', zorder=3, error_kw=dict(capsize=6, capthick=3, elinewidth=3),
                linewidth=0.6)

        # Add a grid to the plot
        plt.grid(True, axis='y', zorder=0)

        # Set range for X and Y axes
        plt.xticks(x, labels)

        # Set y_max and step
        _, _, _, y2 = plt.axis()
        div = find_div(y2)
        y_max = round(y2 / div, 1)
        step = round_down(y_max / 3, 1)
        # plt.yticks(np.arange(0, y_max, step=step))
        plt.yticks(np.arange(0, 1.2, step=0.25))

        # Set font sizes of axes and their ticks
        plt.xlabel('Experiment', fontsize=26)
        plt.ylabel('DTW distance', fontsize=26)
        plt.tick_params(axis='both', labelsize=22)

        # Have ticks on left and right side of Y-axis
        plt.gca().yaxis.set_ticks_position('both')

        # Ticks inside the plot look nicer IMHO
        plt.tick_params(axis='y', direction='in')

        # Set up a legend
        plt.legend(bbox_to_anchor=(0., 0.817, 1.294, .2), loc='center', ncol=2, borderaxespad=0.,
                   fontsize=21, handlelength=1, handletextpad=0.2, columnspacing=0.6)

        # Save plots (let's do both PDF and SVG)
        to_save = os.path.join(save_path, f'{caption}_{modality}')
        plt.savefig(to_save + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.savefig(to_save + '.svg', format='svg', dpi=1000, bbox_inches='tight')
        plt.close('all')


def example_entropy(high_ent_data=None, low_ent_data=None, bins=None, save_path=''):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    if bins is None:
        bins = np.arange(8, 16)

    if high_ent_data is None:
        high_ent_data = np.array([8, 9, 10, 11, 11, 11, 12, 12, 14, 15])

    if low_ent_data is None:
        low_ent_data = np.append(np.full(5, 10), np.append(np.full(4, 11), np.full(1, 12)))

    bin_size = len(bins) - 1

    # Calculate the entropy for both low entropy series and high entropy series
    ent_high = scipy_entropy(probability_hist(high_ent_data, bins)) / np.log2(bin_size)
    ent_low = scipy_entropy(probability_hist(low_ent_data, bins)) / np.log2(bin_size)
    print(f'entropy :{ent_high}')
    print(f'entropy :{ent_low}')

    # Adjust font for 10^x
    plt.rc('font', size=18)

    # Figure size
    plt.figure(figsize=(6.5, 4))

    # Plot distribution and its frame
    kde = stats.gaussian_kde(high_ent_data)
    plt.hist(high_ent_data, bins=bins, edgecolor='black', density=True, zorder=3)
    plt.plot(bins, kde(bins), linewidth=6, color='red', zorder=4)

    # Set axes name sizes and ticks
    plt.xlabel('Bins', fontsize=26)
    plt.ylabel('Probability', fontsize=26)
    plt.tick_params(axis='both', labelsize=22)
    plt.xticks(np.arange(-15, 16, 15))
    # Add 10^x for ticks (Y-axis)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, -1), useMathText=True)
    # plt.yticks(np.arange(0, 0.4, 0.1))
    plt.yticks(np.arange(0, 0.091, 0.03))
    plt.grid(True, axis='y', zorder=0)

    # Have ticks on left and right side of Y-axis
    plt.gca().yaxis.set_ticks_position('both')

    # Ticks inside the plot look nicer IMHO
    plt.tick_params(axis='y', direction='in')

    # Save plots (let's do both PDF and SVG)
    plt.savefig(os.path.join(save_path, 'high_entropy') + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'high_entropy') + '.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.close()

    # Figure size
    plt.figure(figsize=(6.5, 4))

    # Plot distribution and its frame
    kde = stats.gaussian_kde(low_ent_data)
    plt.hist(low_ent_data, bins=bins, edgecolor='black', density=True, zorder=3)
    plt.plot(bins, kde(bins), linewidth=6, color='red', zorder=4)

    # Set axes name sizes and ticks
    plt.xlabel('Bins', fontsize=26)
    plt.ylabel('Probability', fontsize=26)
    plt.tick_params(axis='both', labelsize=22)
    plt.xticks(np.arange(-15, 16, 15))
    plt.yticks(np.arange(0, 0.31, 0.1))
    plt.grid(True, axis='y', zorder=0)

    # Have ticks on left and right side of Y-axis
    plt.gca().yaxis.set_ticks_position('both')

    # Ticks inside the plot look nicer IMHO
    plt.tick_params(axis='y', direction='in')

    # Save plots (let's do both PDF and SVG)
    plt.savefig(os.path.join(save_path, 'low_entropy') + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'low_entropy') + '.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.close()
