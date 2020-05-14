#!/bin/bash

cd /content/
git clone https://github.com/a-doering/tracker_w_correlation_motion_model

export HOME='/content/tracker_w_correlation_motion_model'
cd

add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.7
apt install virtualenv

virtualenv -p python3.7 ~/.venv
source ~/.venv/bin/activate

pip install -r requirements.txt
pip install -e .

