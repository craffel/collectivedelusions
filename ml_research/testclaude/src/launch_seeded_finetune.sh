#!/bin/bash
# Fine-tune 8 tasks for each of 3 seeds, one seed at a time, 8 tasks in parallel on 8 GPUs.
set -e
ROOT=/fsx/craffel/collectivedelusions/ml_research/testclaude
cd $ROOT
mkdir -p logs

TASKS=(MNIST CIFAR10 CIFAR100 SVHN FashionMNIST EuroSAT GTSRB DTD)
EPOCHS=(3 5 5 3 3 5 5 10)
SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    OUT=$ROOT/checkpoints_seed${SEED}
    mkdir -p $OUT
    echo "===== Seed $SEED -> $OUT ====="
    for i in "${!TASKS[@]}"; do
        T=${TASKS[$i]}
        E=${EPOCHS[$i]}
        if [ -f "$OUT/${T}.pt" ]; then
            echo "skip $T (exists)"
            continue
        fi
        echo "Launching $T on GPU $i (seed=$SEED, $E epochs)..."
        python $ROOT/src/finetune.py \
            --task $T --epochs $E --gpu $i --lr 1e-5 --batch_size 128 \
            --seed $SEED --out_dir $OUT \
            > $ROOT/logs/finetune_${T}_seed${SEED}.log 2>&1 &
    done
    wait
    echo "seed $SEED done"
done
echo "All seed fine-tuning complete."
