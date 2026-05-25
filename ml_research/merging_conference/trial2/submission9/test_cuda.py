import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    # Try a simple conv
    conv = torch.nn.Conv2d(3, 64, 3).cuda()
    x = torch.randn(2, 3, 32, 32).cuda()
    try:
        out = conv(x)
        print("Conv success!")
    except Exception as e:
        print("Conv failed with exception:", e)
        print("Attempting to disable cuDNN...")
        torch.backends.cudnn.enabled = False
        try:
            out2 = conv(x)
            print("Conv success after disabling cuDNN!")
        except Exception as e2:
            print("Conv still failed after disabling cuDNN:", e2)
