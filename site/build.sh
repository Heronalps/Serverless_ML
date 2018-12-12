#!/bin/bash

cd ~/Temp_Predictor/site
virtualenv venv --python=python3
source venv/bin/activate
pip install -r ./requirements.txt

current_path=$PWD
path1="$PWD/venv/lib64/python3.6/dist-packages"
path2="$PWD/venv/lib/python3.6/dist-packages"
rm -f lambda_function.zip
zip -rj9 lambda_function.zip ./predictor/predictor.py 

cd $path1
zip -ur $current_path/lambda_function.zip numpy/ matplotlib/ pandas/ scipy/ sklearn/

cd $path2
zip -ur $current_path/lambda_function.zip pyparsing.py six.py pytz/

cd $current_path
echo 'Complete packaging lambda function'

aws s3api put-object --bucket temp-predictor --key lambda_function.zip --body lambda_function.zip
echo 'Uploaded to s3 bucket'