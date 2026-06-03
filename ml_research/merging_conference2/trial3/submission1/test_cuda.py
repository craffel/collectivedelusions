import torch
import traceback

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA Version:", torch.version.cuda)
    try:
        print("cuDNN Version:", torch.backends.cudnn.version())
    except Exception as e:
        print("Error getting cuDNN version:", e)

    # Test 1: Simple tensor on GPU
    try:
        x = torch.rand(2, 2).cuda()
        print("Tensor allocation on CUDA successful.")
    except Exception as e:
        print("Tensor allocation on CUDA failed:")
        traceback.print_exc()

    # Test 2: Standard conv2d on GPU
    try:
        y = torch.nn.functional.conv2d(torch.zeros(1, 1, 32, 32).cuda(), torch.zeros(1, 1, 3, 3).cuda())
        print("Standard Conv2d on CUDA successful.")
    except Exception as e:
        print("Standard Conv2d on CUDA failed:")
        traceback.print_exc()

    # Test 3: Standard conv2d with cuDNN disabled
    try:
        torch.backends.cudnn.enabled = False
        y = torch.nn.functional.conv2d(torch.zeros(1, 1, 32, 32).cuda(), torch.zeros(1, 1, 3, 3).cuda())
        print("Conv2d with cuDNN disabled on CUDA successful.")
    except Exception as e:
        print("Conv2d with cuDNN disabled on CUDA failed:")
        traceback.print_exc()
