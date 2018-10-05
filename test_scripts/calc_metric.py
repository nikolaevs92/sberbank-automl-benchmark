import argparse
import os
import pandas as pd
from datetime import datetime

from sklearn.metrics import roc_auc_score, mean_squared_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true-dir-path', required=True)
    parser.add_argument('--predict-dir-path', required=True)
    parser.add_argument('--model-name', required=True)
    args = parser.parse_args()

    true_dir_path = args.true_dir_path
    predict_dir_path = args.predict_dir_path
    true_csv = 'test-target.csv'
    prediction_csv = 'prediction.csv'

    model_name = args.model_name

    results = {
        'model_name': model_name,
        'datetime': datetime.now()
        }

    for check in os.listdir('data'):
        # load datasets
        true_target = pd.read_csv(f'{true_dir_path}/{check}/{true_csv}')
        predict_target = pd.read_csv(f'{predict_dir_path}/{check}/{prediction_csv}')

        #calculate metric
        if check[-1] == "r":
            result = mean_squared_error(true_target.target, predict_target.prediction)**0.5
        elif check[-1] == "c":
            result = roc_auc_score(true_target.target, predict_target.prediction)
        results[check] = result

    print(results)
    # save to csv
    if os.stat("test_result.csv").st_size == 0:
        pd.DataFrame([results]).to_csv('test_result.csv', index=False)
    else:
        pd.read_csv('test_result.csv', index_col='datetime'
        ).append(pd.DataFrame([results])).to_csv('test_result.csv', index='datetime')
    