import pandas as pd
from helpers import *


n = 1000

bootstrap_results = pd.DataFrame()
# Baseline vs Grouped - assessing grouping of codes
for i in range(n):
    bootstrap_results = pd.concat([bootstrap_results, one_run('baseline', 'grouped')])

bootstrap_results.to_csv('../../results/validation/bootstrap_results.csv', index=False)
calculate_stats(bootstrap_results).to_csv('../../results/validation/bootstrap_stats.csv', index=False)
