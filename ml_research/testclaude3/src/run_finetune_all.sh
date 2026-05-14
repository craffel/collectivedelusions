#!/bin/bash
# Launch fine-tuning on all 7 tasks in parallel across 7 GPUs.
set -u
cd /fsx/craffel/collectivedelusions/ml_research/testclaude3
export PATH=/fsx/craffel/collectivedelusions/ml_research/testclaude3/bin:$PATH

TASKS=(MNIST SVHN CIFAR10 CIFAR100 EuroSAT GTSRB DTD)

pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  echo "Launching $t on GPU $i"
  uv run python -m src.finetune --task "$t" --gpu "$i" --epochs 3 \
    --lr 1e-5 --batch-size 128 --num-workers 4 \
    --max-train 20000 --max-test 4000 \
    > "logs/ft_${t}.stdout" 2>&1 &
  pids+=($!)
done
echo "Started ${#pids[@]} jobs: ${pids[*]}"
wait "${pids[@]}"
echo "All fine-tuning done"
