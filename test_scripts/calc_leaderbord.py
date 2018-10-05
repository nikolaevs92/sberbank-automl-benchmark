import argparse
import os
import pandas as pd
import json
from datetime import datetime

from sklearn.metrics import roc_auc_score, mean_squared_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-csv', required=True)
    parser.add_argument('--leaderbord-csv', required=True)
    args = parser.parse_args()

    results_csv = args.results_csv
    leaderbord_csv = args.leaderbord_csv

    results = pd.read_csv(results_csv)
    with open('baseline.json', 'r') as file:
        baseline = json.load(file)

    checks = list(baseline.keys())
    checks.remove('model_name')

    for key in checks:
        minus_baseline =  key + '-baseline'
        if key[-1] == 'r':
            results[minus_baseline] = - results[key] + baseline[key]
            results[key + '-score'] = results[minus_baseline]/results[minus_baseline].max()
            results[key + '-idealscore'] = results[minus_baseline]/baseline[key]
        elif key[-1] == 'c':
            results[minus_baseline] = results[key] - baseline[key]
            results[key + '-score'] = results[minus_baseline]/results[minus_baseline].max()
            results[key + '-idealscore'] = results[minus_baseline]/(1-baseline[key])

    results['score'] = results[[key+'-score' for i in checks]].values.sum(axis=1)
    results['ideal_score'] = results[[key+'-idealscore' for i in checks]].values.sum(axis=1)
    
    # save to csv
    results[['model_name', 'datetime', 'score', 'ideal_score']].to_csv(leaderbord_csv)
    