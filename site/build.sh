#!/bin/bash

cd Temp_Predictor/site
virtualenv venv --python=python3
source venv/bin/activate
pip install -r ./requirements.txt

current_path=$PWD
path="$PWD/venv/lib/python3.6/site-packages"
rm -f lambda_function.zip
zip -r9 lambda_function.zip ./predictor/predictor.py 

cd $path
echo $PWD
zip -ur $current_path/lambda_function.zip numpy/ matplotlib/ pandas/ scipy/ sklearn/
cd $current_path
echo 'Complete packaging lambda function'
