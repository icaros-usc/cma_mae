import re
import csv
import glob
import numpy as np
import pandas as pd
from os import path

import seaborn as sns
import matplotlib.pyplot as plt

# Change for each experiment.
#exp_name = 'lin_proj'

#experiment_folder = f"../{exp_name}/results_arch_res/sphere_logs/"
#fig_name = "Linear Projection (sphere)"

#experiment_folder = f"../{exp_name}/results_arch_res/rastrigin_logs/"
#fig_name = "Linear Projection (Rastrigin)"

#experiment_folder = f"../{exp_name}/results_arch_res/plateau_logs/"
#fig_name = "Linear Projection (plateau)"

exp_name = 'arm'
experiment_folder = f"../{exp_name}/results_arch_res/"

# Change for each algorithm.
algorithms = [
    ('cma_mae_100_0.01', 'CMA-MAE'),
    ('cma_me_imp_100', 'CMA-ME'),
    ('map_elites_line_100', 'MAP-Elites (line)'),
    ('map_elites_100', 'MAP-Elites'),
]

# How many trials?
total_trials = 25

# Add a legend?
have_legend = True

def pandas_data_frame(algorithm_id, algorithm_name):

    total_summary_filename = "summary.csv"
    summary_path = experiment_folder + "{}/trial_*/summary.csv"
    summary_path = summary_path.format(algorithm_id)

    all_data = [['Algorithm', 'Resolution', 'QD-Score', 'Coverage', 'Maximum', 'Average']]
    for summary_file_path in glob.glob(summary_path):
        head, filename = path.split(summary_file_path)
        head, trial_name = path.split(head)
        head, algo_name = path.split(head)
        _, trial_id = re.split('trial_', trial_name)
        trial_id = int(trial_id)

        index = 0.0
        if total_trials > 1:
            index = trial_id / (total_trials-1.0)

        min_count = 50
        max_count = 500
        resolution = int(index * (max_count - min_count) + min_count + 1e-9)

        with open(summary_file_path) as summary_file:
            all_lines = list(csv.reader(summary_file))

            datum = [float(v) for v in all_lines[-1][1:]]
            datum = [algorithm_name, resolution] + datum
            all_data.append(datum)

    print(all_data[1:])
    df = pd.DataFrame(np.array(all_data[1:]),
                 columns=all_data[0])
    df['Resolution'] = df['Resolution'].astype('int')
    df['QD-Score'] = df['QD-Score'].astype('float')
    df['Coverage'] = df['Coverage'].astype('float')
    return df


df_list = []

for algorithm_id, algorithm_name in algorithms:
    print(algorithm_id, algorithm_name)
    df_i = pandas_data_frame(algorithm_id, algorithm_name)
    df_list.append(df_i)
df = pd.concat(df_list)

sns.set(font_scale=2.4)
with sns.axes_style("white"):
    sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})

    p1 = sns.lineplot(
            data=df, 
            x='Resolution', 
            y='QD-Score',
            #y='Coverage',
            hue="Algorithm",
            legend=have_legend,
    )        

    for line in p1.get_lines():
        line.set_linewidth(4.0)

    plt.xticks([50, 500])
    plt.yticks([0, 85])
    
    if have_legend:
        legend = plt.legend(loc="lower left", frameon=False, prop={'size': 20})
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        frame = legend.get_frame()
        frame.set_facecolor('white')


    #plt.title(fig_name, y=1.1, fontsize=20)
    plt.tight_layout()

    #plt.show()

    p1.figure.savefig("res_exp.pdf", bbox_inches='tight', dpi=100)
