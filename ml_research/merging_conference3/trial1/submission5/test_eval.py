import sys
import os

# Add task_vectors to path
sys.path.append(os.path.abspath('task_vectors'))

import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments

# Mock command line arguments
sys.argv = [
    'test_eval.py',
    '--data-location', 'data',
    '--model', 'ViT-B-32',
    '--save', 'task_vectors_checkpoints/ViT-B-32',
    '--eval-datasets', 'MNIST'
]

args = parse_arguments()
pretrained_checkpoint = 'task_vectors_checkpoints/ViT-B-32/zeroshot.pt'
finetuned_checkpoint = 'task_vectors_checkpoints/ViT-B-32/MNIST/finetuned.pt'

print("Loading task vector...")
task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
print("Applying task vector with scaling coef 0.5...")
image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.5)

print("Evaluating...")
eval_single_dataset(image_encoder, 'MNIST', args)
print("Sanity check completed successfully!")
