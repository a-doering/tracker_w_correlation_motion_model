
cd /content
source .venv/bin/activate
export HOME='/content/tracker_w_correlation_motion_model'
cd

git config core.filemode false --replace-all

alias l='ls -lrth'
alias test_tracktor='python ~/experiments/scripts/test_tracktor.py'
alias train_correlation='python ~/experiments/scripts/train_correlation.py'
alias gst='git status -sb'
alias clean='rm -rf .cache/ .nv/'
