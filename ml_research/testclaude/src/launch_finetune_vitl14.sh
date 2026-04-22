#!/bin/bash
set -e
ROOT=/fsx/craffel/collectivedelusions/ml_research/testclaude
cd $ROOT
mkdir -p logs
export HF_HOME=$ROOT/hf_cache

TASKS=(MNIST CIFAR10 CIFAR100 SVHN FashionMNIST EuroSAT GTSRB DTD)
EPOCHS=(2 3 3 2 2 3 3 6)

for i in "${!TASKS[@]}"; do
    T=${TASKS[$i]}
    E=${EPOCHS[$i]}
    SAVE_BASE=""
    if [ $i -eq 0 ]; then SAVE_BASE="--save_base"; fi
    echo "Launching ViT-L/14 $T on GPU $i ($E epochs)..."
    python $ROOT/src/finetune.py \
        --task $T --model ViT-L-14 --pretrained openai \
        --epochs $E --gpu $i --lr 1e-5 --batch_size 64 \
        --out_dir $ROOT/checkpoints_vitl14 \
        $SAVE_BASE \
        > $ROOT/logs/finetune_vitl14_$T.log 2>&1 &
done
wait
echo "All ViT-L/14 fine-tuning complete."
