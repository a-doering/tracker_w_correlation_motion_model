
cd /content
source .venv/bin/activate
export HOME='/content/tracker_w_correlation_motion_model'
cd

git config core.filemode false --replace-all

alias l='ls -lrth'
alias run_test='python ~/experiments/scripts/test_tracktor.py'
