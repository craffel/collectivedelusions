#!/bin/bash
cd /fsx/craffel/collectivedelusions/ml_research/testclaude3
export PATH=/fsx/craffel/collectivedelusions/ml_research/testclaude3/bin:$PATH
TASKS=(MNIST SVHN CIFAR10 CIFAR100 EuroSAT GTSRB DTD)
GPUS=(1 2 3 4 5 6 7)
SEED=2
mkdir -p checkpoints_seed${SEED}
pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  g=${GPUS[$i]}
  uv run python -m src.finetune --task "$t" --gpu "$g" --epochs 3 \
    --lr 1e-5 --batch-size 128 --num-workers 4 \
    --max-train 20000 --max-test 4000 \
    --seed "$SEED" \
    --out-dir checkpoints_seed${SEED} \
    > "logs/ft_seed${SEED}_${t}.stdout" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo done
