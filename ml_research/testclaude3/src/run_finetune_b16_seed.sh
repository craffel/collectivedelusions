#!/bin/bash
# Fine-tune CLIP ViT-B/16 with a specific seed.
set -u
cd /fsx/craffel/collectivedelusions/ml_research/testclaude3
export PATH=/fsx/craffel/collectivedelusions/ml_research/testclaude3/bin:$PATH
export CLIP_ARCH=ViT-B-16-quickgelu
export CLIP_PRETRAINED=openai

SEED=${1:-1}
TASKS=(MNIST SVHN CIFAR10 CIFAR100 EuroSAT GTSRB DTD)
OUTDIR=checkpoints_b16_seed${SEED}

mkdir -p "$OUTDIR" logs

pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  uv run python -m src.finetune --task "$t" --gpu "$i" --epochs 3 \
    --lr 1e-5 --batch-size 64 --num-workers 4 \
    --max-train 16000 --max-test 4000 \
    --seed "$SEED" \
    --out-dir "$OUTDIR" \
    > "logs/ft_b16_seed${SEED}_${t}.stdout" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "All B/16 seed=$SEED fine-tuning done"
