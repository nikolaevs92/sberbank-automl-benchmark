#!/bin/bash
automl_path=$1
leaderbord_csv=$2
data_path='data/'
temp_dir='temp'


mkdir ${temp_dir}

for check in $(sudo ls data | egrep '\_r')
do
mkdir ${temp_dir}/${check}
echo ${check} start train
python ${automl_path}/train.py --train-csv ${data_path}/${check}/train.csv --model-dir ${temp_dir}/${check} --mode regression
echo ${check} start predict 
python ${automl_path}/predict.py --test-csv ${data_path}/${check}/test.csv --prediction-csv ${temp_dir}/${check}/prediction.csv --model-dir ${temp_dir}/${check}
done

for check in $(sudo ls data | egrep '\_c')
do
mkdir ${temp_dir}/${check}
echo ${check} start train
python ${automl_path}/train.py --train-csv ${data_path}/${check}/train.csv --model-dir ${temp_dir}/${check} --mode classification
echo ${check} start predict 
python ${automl_path}/predict.py --test-csv ${data_path}/${check}/test.csv --prediction-csv ${temp_dir}/${check}/prediction.csv --model-dir ${temp_dir}/${check}
done
echo start calculate metrics 

python test_scripts/calc_metric.py --true-dir-path ${data_path} --predict-dir-path ${temp_dir} --model-name ${automl_path}

python test_scripts/calc_leaderbord.py --results-csv test_result.csv --leaderbord-csv ${leaderbord_csv}

rm -r temp/