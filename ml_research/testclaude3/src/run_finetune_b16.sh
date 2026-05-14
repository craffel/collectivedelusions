#!/bin/bash
# Fine-tune CLIP ViT-B/16 visual encoder on each task in parallel across 7 GPUs.
set -u
cd /fsx/craffel/collectivedelusions/ml_research/testclaude3
export PATH=/fsx/craffel/collectivedelusions/ml_research/testclaude3/bin:$PATH
export CLIP_ARCH=ViT-B-16-quickgelu
export CLIP_PRETRAINED=openai

TASKS=(MNIST SVHN CIFAR10 CIFAR100 EuroSAT GTSRB DTD)
OUTDIR=checkpoints_b16

mkdir -p "$OUTDIR" logs

pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  echo "Launching $t on GPU $i (B/16)"
  uv run python -m src.finetune --task "$t" --gpu "$i" --epochs 3 \
    --lr 1e-5 --batch-size 64 --num-workers 4 \
    --max-train 16000 --max-test 4000 \
    --seed 0 \
    --out-dir "$OUTDIR" \
    > "logs/ft_b16_${t}.stdout" 2>&1 &
  pids+=($!)
done
echo "Started ${#pids[@]} jobs: ${pids[*]}"
wait "${pids[@]}"
echo "All B/16 fine-tuning done"
