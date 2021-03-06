
export HOME='/content'
cd
source .venv/bin/activate
cp ./tracker_w_correlation_motion_model/.gitconfig ./

cd ./tracker_w_correlation_motion_model

git config core.filemode false --replace-all

alias l='ls -lrth'
alias version='python -V && nvidia-smi -L'
alias test_tracktor='version && python ~/tracker_w_correlation_motion_model/experiments/scripts/test_tracktor.py'
alias train_correlation='version && python ~/tracker_w_correlation_motion_model/experiments/scripts/train_correlation.py'
alias gst='git status -sb'
alias clean='rm -rf .cache/ .nv/'
alias install_venv='pip install -r requirements.txt && pip install spatial-correlation-sampler && pip install -e .'
alias clean_training='rm -rf output/tracktor/correlation/test/; rm -rf tensorboard/tracker/'
