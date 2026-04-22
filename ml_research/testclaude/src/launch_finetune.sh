#!/bin/bash
set -e
ROOT=/fsx/craffel/collectivedelusions/ml_research/testclaude
cd $ROOT
mkdir -p logs
export HF_HOME=$ROOT/hf_cache

TASKS=(MNIST CIFAR10 CIFAR100 SVHN FashionMNIST EuroSAT GTSRB DTD)
EPOCHS=(3 5 5 3 3 5 5 10)

for i in "${!TASKS[@]}"; do
    T=${TASKS[$i]}
    E=${EPOCHS[$i]}
    SAVE_BASE=""
    if [ $i -eq 0 ]; then SAVE_BASE="--save_base"; fi
    echo "Launching $T on GPU $i ($E epochs)..."
    python $ROOT/src/finetune.py \
        --task $T --epochs $E --gpu $i --lr 1e-5 --batch_size 128 \
        $SAVE_BASE \
        > $ROOT/logs/finetune_$T.log 2>&1 &
done
wait
echo "All fine-tuning complete."
