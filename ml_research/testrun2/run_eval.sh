#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cusparselt/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cufft/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/curand/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cusolver/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/cusparse/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib:/fsx/craffel/miniconda3/lib/python3.12/site-packages/nvidia/nvtx/lib

echo "Starting evaluation script..."
python evaluate_merges.py
