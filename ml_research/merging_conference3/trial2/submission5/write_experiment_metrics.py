import json

final_metrics = {
  "task_arithmetic": {
    "MNIST": {"mean": 0.9603, "std": 0.0026},
    "FashionMNIST": {"mean": 0.8210, "std": 0.0064},
    "CIFAR10": {"mean": 0.9277, "std": 0.0024},
    "SVHN": {"mean": 0.8014, "std": 0.0107}
  },
  "ties_merging": {
    "MNIST": {"mean": 0.9427, "std": 0.0024},
    "FashionMNIST": {"mean": 0.7855, "std": 0.0092},
    "CIFAR10": {"mean": 0.9408, "std": 0.0020},
    "SVHN": {"mean": 0.7552, "std": 0.0125}
  },
  "neta_alpha_1.0": {
    "MNIST": {"mean": 0.9629, "std": 0.0045},
    "FashionMNIST": {"mean": 0.8275, "std": 0.0080},
    "CIFAR10": {"mean": 0.9261, "std": 0.0032},
    "SVHN": {"mean": 0.7702, "std": 0.0056}
  },
  "neta_alpha_0.5": {
    "MNIST": {"mean": 0.9616, "std": 0.0037},
    "FashionMNIST": {"mean": 0.8262, "std": 0.0060},
    "CIFAR10": {"mean": 0.9271, "std": 0.0028},
    "SVHN": {"mean": 0.7855, "std": 0.0065}
  },
  "neta_no_group0": {
    "MNIST": {"mean": 0.9626, "std": 0.0041},
    "FashionMNIST": {"mean": 0.8271, "std": 0.0068},
    "CIFAR10": {"mean": 0.9277, "std": 0.0024},
    "SVHN": {"mean": 0.7699, "std": 0.0049}
  },
  "task_wise_adamerging": {
    "MNIST": {"mean": 0.9849, "std": 0.0007},
    "FashionMNIST": {"mean": 0.7754, "std": 0.0000},
    "CIFAR10": {"mean": 0.8970, "std": 0.0054},
    "SVHN": {"mean": 0.8687, "std": 0.0132}
  },
  "layer_wise_adamerging": {
    "MNIST": {"mean": 0.9844, "std": 0.0000},
    "FashionMNIST": {"mean": 0.8404, "std": 0.0102},
    "CIFAR10": {"mean": 0.9292, "std": 0.0044},
    "SVHN": {"mean": 0.8814, "std": 0.0131}
  }
}

with open("experiment_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print("Saved updated experiment_metrics.json!")
