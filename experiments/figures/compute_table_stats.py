# A script that computes the averages for all statistics across all
# trials for each algorithm in an experiment.
#
# Run the ``summarize_final.py'' script to get a summary file first.

import pandas as pd

summary_filename = 'summary.csv'

algorithm_ids = [
   'cma_me_io_100',
]

df = pd.read_csv(summary_filename)
print(df)

df['QD-Score (avg)'] = df.groupby('Algorithm')['QD-Score'].transform('mean')
df['Coverage (avg)'] = df.groupby('Algorithm')['Coverage'].transform('mean')

for algo in algorithm_ids:
    print(algo, df[df['Algorithm'] == algo])
