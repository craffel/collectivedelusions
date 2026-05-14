#!/bin/bash
# Re-fine-tune on a different seed to assess result stability.
set -u
cd /fsx/craffel/collectivedelusions/ml_research/testclaude3
export PATH=/fsx/craffel/collectivedelusions/ml_research/testclaude3/bin:$PATH

SEED=${1:-1}
TASKS=(MNIST SVHN CIFAR10 CIFAR100 EuroSAT GTSRB DTD)
OUTDIR=checkpoints_seed${SEED}

mkdir -p "$OUTDIR"

pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  echo "Launching $t on GPU $i with seed $SEED"
  uv run python -m src.finetune --task "$t" --gpu "$i" --epochs 3 \
    --lr 1e-5 --batch-size 128 --num-workers 4 \
    --max-train 20000 --max-test 4000 \
    --seed "$SEED" \
    --out-dir "$OUTDIR" \
    > "logs/ft_seed${SEED}_${t}.stdout" 2>&1 &
  pids+=($!)
done
echo "Started ${#pids[@]} jobs: ${pids[*]}"
wait "${pids[@]}"
echo "All seed=$SEED fine-tuning done"
