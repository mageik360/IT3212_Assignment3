#!/usr/bin/env bash
# Rerun every model in sequence on one machine and save all output under result/.
#
# Run from anywhere:
#     bash scripts/rerun_all.sh
#     PYTHON=python3 bash scripts/rerun_all.sh   # if "python" is not the right interpreter
#
# Steps run one after another, never in parallel, so the reported times are not
# distorted. The neural networks use the GPU automatically when available.
# Console output of each step goes to result/rerun/<step>.log, and
# result/rerun/chain.log records when each step starts and ends. The scripts also
# write their usual files to result/ and figures/. Stops at the first failing step.
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export PYTHONUNBUFFERED=1
LOG_DIR=result/rerun
mkdir -p "$LOG_DIR"

run() {
    local name=$1; shift
    echo "[$(date '+%F %T')] start $name" | tee -a "$LOG_DIR/chain.log"
    "$PYTHON" -u "$@" > "$LOG_DIR/$name.log" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] end $name (exit $code)" | tee -a "$LOG_DIR/chain.log"
    return $code
}

run decision_tree src/decision_tree.py || exit 1
run logistic_regression src/logistic_regression.py || exit 1
# Tuned shallow network, refitted on train + validation
run neural_network src/neural_network.py --refit || exit 1
# The selected settings again without refit, for the no-refit test metrics
cfg=result/neural_network/selected_config.json
args=$("$PYTHON" -c "import json; c = json.load(open('$cfg')); print(f\"--hidden-dim {c['hidden_dim']} --learning-rate {c['learning_rate']} --weight-decay {c['weight_decay']}\")")
run neural_network_norefit src/neural_network.py --no-tune $args \
    --figures-dir figures/neural_network/no_refit --results-dir result/neural_network/no_refit || exit 1
run bagging_boosting src/bagging_boosting.py || exit 1
# 30-trial search, then 3 seeds each of the shallow, starting and tuned deep networks
run advanced_neural_network src/advanced_neural_network.py --mode compare --epochs 50 --trials 30 || exit 1
echo "[$(date '+%F %T')] all done" | tee -a "$LOG_DIR/chain.log"
