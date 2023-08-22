import os
import json
import plot_metrics
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def list_files(file_path, file_format):
    return glob(os.path.join(os.path.join(file_path, '**'), file_format), recursive=True)


def get_name(label):
    names = ['attacker_JBL-const',                      # 0
             'attacker_JBL-stair',                      # 1
             'entropy_attacker_JBL-JBL_dist0m',         # 2
             'entropy_attacker_JBL-JBL_dist0m-fast',    # 3
             'entropy_attacker_JBL-JBL_dist0.5m-fast',  # 4
             'entropy_attacker_JBL-JBL_dist1m-fast',    # 5
             'entropy_attacker_JBL-JBL_dist2m-fast',    # 6
             'entropy_attacker_JBL-robot',              # 7
             'entropy_attacker_cheap-JBL']              # 8

    if label == names[0]:
        return 'AA\nConst.'
    if label == names[1]:
        return 'AA\nStair.'
    if label == names[2]:
        return 'AA + H\nNorm.'
    if label == names[3]:
        # return 'AA + H\nFast'
        return 'AA + H\n0m'
    if label == names[4]:
        return 'AA + H\n(f_0.5m)'
    if label == names[5]:
        return 'AA + H\n(f_1m)'
    if label == names[6]:
        return 'AA + H\n2m'
    if label == names[7]:
        return 'AA + H\nFl.5-S5'
    if label == names[8]:
        return 'AA + H\nA2-Fl.5'


def get_followup_names():
    return ['attacker_JBL-const',
            'attacker_JBL-stair',
            'entropy_attacker_JBL-JBL_dist0m',
            'entropy_attacker_JBL-JBL_dist0m-fast',
            'entropy_attacker_JBL-JBL_dist0.5m-fast',
            'entropy_attacker_JBL-JBL_dist1m-fast',
            'entropy_attacker_JBL-JBL_dist2m-fast',
            'entropy_attacker_JBL-robot',
            'entropy_attacker_cheap-JBL']


def get_followup_labels(labels):
    new_labels = []
    for label in labels:
        new_labels.append(get_name(label))

    return new_labels, np.arange(len(new_labels))


def generate_distance_plot_experiment_wise(file_path, sim_metric='soundProofXcorr', exclude_list=[],
                                           arrange_order=False, is_follow_up=False):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    # Read results from json files
    results = read_json_files(file_path, sim_metric)

    # Define vars we need
    labels = []
    audio_sim = {}
    chunks = set()

    # Iterate over sub-experiments (e.g., 'attacker')
    for experiment in results:
        if exclude_list.__contains__(experiment):
            continue
        labels.append(experiment)
        audio_sim[experiment] = {}

        # Iterate over chunks (e.g., 10 sec)
        for chunk in results[experiment]:
            if chunk == 'path':
                continue
            chunks.add(chunk)

            if not audio_sim[experiment].__contains__(chunk):
                audio_sim[experiment][chunk] = {}
                audio_sim[experiment][chunk]['co'] = list()
                audio_sim[experiment][chunk]['nonco'] = list()

            # Iterate over individual sensors
            for sensor in results[experiment][chunk]:
                for xdevice in results[experiment][chunk][sensor]:
                    for x in results[experiment][chunk][sensor][xdevice]['results']:
                        if sim_metric == 'timeFreqDistance':
                            data = results[experiment][chunk][sensor][xdevice]['results'][x]['time_freq_dist']
                        else:
                            data = np.mean(list(results[experiment][chunk][sensor][xdevice]['results'][x]['xcorr_freq_bands'].values())[:20])
                        if sensor.__contains__('2') or xdevice.__contains__('2'):
                            audio_sim[experiment][chunk]['nonco'].append(data)
                        else:
                            audio_sim[experiment][chunk]['co'].append(data)

    # Get chunk sizes in a list and sort it
    chunks = list(chunks)
    chunks.sort()

    # These labels are for 'Injection speed and hardware' plot, i.e., Figure 7b
    # labels = ['entropy_attacker_JBL-JBL_dist0m', 'entropy_attacker_JBL-JBL_dist0m-fast',
    #           'entropy_attacker_cheap-JBL', 'entropy_attacker_JBL-robot']

    # Iterate over chunks
    for chunk in chunks:
        mean_coloc = []
        mean_non_coloc = []
        std_coloc = []
        std_non_coloc = []

        # If we consider all experiments in the follow-up, let's rearrange bars a bit
        followup_flag = False
        if is_follow_up:
            if sorted(labels) == sorted(get_followup_names()):
                labels = get_followup_names()
                followup_flag = True

        for experiment in labels:
            mean_coloc.append(np.mean(audio_sim[experiment][chunk]['co']))
            std_coloc.append(np.std(audio_sim[experiment][chunk]['co']))
            mean_non_coloc.append(np.mean(audio_sim[experiment][chunk]['nonco']))
            std_non_coloc.append(np.std(audio_sim[experiment][chunk]['nonco']))

        # Adjust hatch size and color
        plt.rcParams.update({'hatch.color': 'white'})
        plt.rcParams.update({'hatch.linewidth': 3})

        # Color bars nicely
        colors = ['#c14a09', '#06b48b', '#9d5783', '#287c37']

        # Set figure size
        plt.figure(figsize=(6.5, 4))

        # Width of the bars
        width = 0.365

        # Get X-axis labels
        if arrange_order:
            display_labels, x = plot_metrics.get_entropy_labels_and_index(labels)
        elif is_follow_up:
            display_labels, x = get_followup_labels(labels)
        else:
            display_labels = labels
            x = np.arange(len(labels))  # the label locations

        # Create bar plots
        plt.bar(x - width / 2, mean_coloc, width, yerr=std_coloc, label='Coloc.', zorder=3, color=colors[1],
                hatch='/', error_kw=dict(capsize=6, capthick=3, elinewidth=3), linewidth=0.6)

        plt.bar(x + width / 2, mean_non_coloc, width, yerr=std_non_coloc, label='Non-coloc.', zorder=3, color=colors[0],
                hatch='.', error_kw=dict(capsize=6, capthick=3, elinewidth=3), linewidth=0.6)

        # Add a grid to the plot
        plt.grid(True, axis='y', zorder=0)

        # Set Y-axis label
        if sim_metric == 'timeFreqDistance':
            y_label = 'Time-freq. distance'
        else:
            y_label = 'Similarity score'

        # Set range for X and Y axes
        plt.xticks(x, display_labels)

        # Set y_max and step
        _, _, _, y2 = plt.axis()
        div = plot_metrics.find_div(y2)
        y_max = round(y2 / div, 1)
        step = plot_metrics.round_down(y_max / 2.5, 1)
        if sim_metric == 'soundProofXcorr':
            plt.yticks(np.arange(0, 1.2, step=0.25))
        else:
            plt.yticks(np.arange(0, y_max, step=step))

        # Plot stuff for paper or analysis
        if followup_flag:
            plt.tick_params(axis='both', labelsize=12)
            plt.xticks(rotation=30)
        else:
            plt.tick_params(axis='both', labelsize=22)

        # Set font sizes of axes and their ticks
        plt.xlabel('Experiment', fontsize=26)
        plt.ylabel(y_label, fontsize=26)
        plt.tick_params(axis='both', labelsize=22)

        # Have ticks on left and right side of Y-axis
        plt.gca().yaxis.set_ticks_position('both')

        # Ticks inside the plot look nicer IMHO
        plt.tick_params(axis='y', direction='in')

        # Set up a legend
        plt.legend(bbox_to_anchor=(0., 0.817, 1.294, .2), loc='center', ncol=2, borderaxespad=0.,
                   fontsize=21, handlelength=1, handletextpad=0.2, columnspacing=0.6)

        # Save plots
        if sim_metric == 'soundProofXcorr':
            to_save = os.path.join(file_path, 'Audio', 'distance', 'Plot_distance', f'distance_Xcorr_{chunk}')
        else:
            to_save = os.path.join(file_path, 'Audio', 'distance', 'Plot_distance', f'distance_TFD_{chunk}')

        # Path to save plots
        save_path = os.path.join(file_path, 'Audio', 'distance', 'Plot_distance')

        # Create directory for saving plots
        os.makedirs(save_path, exist_ok=True)

        # Save plots (let's do both PDF and SVG)
        plt.savefig(to_save + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.savefig(to_save + '.svg', format='svg', dpi=1000, bbox_inches='tight')
        plt.close('all')


def read_json_files(file_path, sim_metric):
    # Read json files
    files = list_files(file_path, os.path.join(sim_metric, '**', '*.json'))
    files.sort()

    # Dict to store the results
    results = {}

    # Iterate over files
    for file in files:
        device = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
        exp_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))
        chunk_size = os.path.basename(os.path.dirname(file))
        experiment = os.path.basename(exp_path)

        if not results.__contains__(experiment):
            results[experiment] = {}
            results[experiment]['path'] = exp_path

        if not results[experiment].__contains__(chunk_size):
            results[experiment][chunk_size] = {}

        xdevice = os.path.basename(file).replace('.json', '')

        # Read json file data
        with open(file, 'r') as json_file:
            if not results[experiment][chunk_size].__contains__(device):
                results[experiment][chunk_size][device] = {}
            results[experiment][chunk_size][device][xdevice] = json.loads(json_file.read())

    return results


def generate_entropy_from_existing_files(file_path, exclude_list=[]):
    # Read result files
    files = glob(os.path.join(file_path, '**', 'audio_entropy.json'), recursive=True)

    # Iterate over result files
    for file in files:
        # Open file for reading
        with open(file, 'r') as json_file:
            # Get experiment name
            experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))

            # Get result data
            data = json.load(json_file)

            # Check if we need to set up a flag for the follow-up experiment
            if experiment.lower().__contains__('new'):
                is_not_followup = False
            else:
                is_not_followup = True

            # Modality name and bin sizes
            modality = ['audio']
            bins = ['10', '15', '15000', '30000', '60000']

            # Path to save plots
            save_path = os.path.join(os.path.dirname(file), 'Plot_entropy')

            # Create directory for saving plots
            os.makedirs(save_path, exist_ok=True)

            # Iterate over modality
            for mod in modality:
                # Iterate over bins
                for b in bins:
                    # Do some plotting
                    plot_metrics.draw_experiment_entropy_bar_plot(data, save_path, f'{exp_name}_entropy_{mod}',
                                                                  b, mod, exclude_list=exclude_list,
                                                                  arrange_order=is_not_followup,
                                                                  is_audio_followup=not is_not_followup)


if __name__ == '__main__':
    # Provide path where results are stored
    # Get the results from: https://zenodo.org/record/8263497
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-results'

    # Get paths for each scenario --> case when we want to cover all scenarios at once
    # experiments = [os.path.join(filepath, 'Home'),
    #                os.path.join(filepath, 'Office'),
    #                os.path.join(filepath, 'Office_new')]

    # Case when we want to plot results for a specific scenario
    experiments = [os.path.join(filepath, 'Home')]

    # Plot a subset of experiments --> 'exclude' is DEFAULT for Home and Office + generating Figure 7a in Office_new:
    # make sure to COMMENT 'labels' in 'generate_distance_plot_experiment_wise'
    # and check that 'if label == names[3]:' in 'get_name' returns the '...(0m)' option
    exclude = ['entropy_attacker_JBL-JBL_dist1m-fast', 'entropy_attacker_JBL-JBL_dist0.5m-fast',
               'entropy_attacker_cheap-JBL', 'entropy_attacker_JBL-robot', 'entropy_attacker_JBL-JBL_dist0m']

    '''
    # Case when we want to plot results for a specific scenario
    experiments = [os.path.join(filepath, 'Office_new')]
    
    # Plot a subset of experiments --> 'exclude' is ONLY for generating Figure 7b in Office_new:
    # make sure to UNCOMMENT 'labels' in 'generate_distance_plot_experiment_wise' 
    # and check that 'if label == names[3]:' in 'get_name' returns the '...(Fast)' option
    exclude = ['entropy_attacker_JBL-JBL_dist1m-fast', 'entropy_attacker_JBL-JBL_dist0.5m-fast', 
               'attacker_JBL-const', 'attacker_JBL-stair', 'entropy_attacker_JBL-JBL_dist2m-fast']
    # '''

    # Iterate over scenarios
    for exp in experiments:
        # Get experiment name
        exp_name = os.path.basename(exp)

        # This is just to set plot labels right
        if exp_name.lower().__contains__('new'):
            is_not_followup = False
            is_followup = True
        else:
            is_not_followup = True
            is_followup = False

        # Xcorr distance bar plot
        generate_distance_plot_experiment_wise(exp, sim_metric='soundProofXcorr', exclude_list=exclude,
                                               arrange_order=is_not_followup, is_follow_up=is_followup)

        # Entropy bar plot
        generate_entropy_from_existing_files(exp, exclude)
