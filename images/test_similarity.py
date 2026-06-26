from PIL import Image
import numpy as np

img1 = Image.open("images_v4/winners_round_0/PCWINNERS-Baby-Animals-1st-WINTER26-960x686.jpg").convert("RGB").resize((1024, 1024))
img2 = Image.open("test_flux2_img2img.jpg").convert("RGB")

arr1 = np.array(img1)
arr2 = np.array(img2)
mse = np.mean((arr1 - arr2) ** 2)
print("MSE:", mse)
