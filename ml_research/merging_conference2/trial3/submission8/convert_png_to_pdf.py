import os
from PIL import Image

def convert_png_to_pdf(png_path, pdf_path):
    print(f"Converting {png_path} to {pdf_path}...")
    if not os.path.exists(png_path):
        print(f"Error: {png_path} does not exist.")
        return False
    im = Image.open(png_path)
    im_rgb = im.convert('RGB')
    im_rgb.save(pdf_path, 'PDF')
    print("Done!")
    return True

convert_png_to_pdf('main_benchmark.png', 'main_benchmark.pdf')
convert_png_to_pdf('sample_efficiency.png', 'sample_efficiency.pdf')
convert_png_to_pdf('distributional_robustness.png', 'distributional_robustness.pdf')
convert_png_to_pdf('results/rtaac_shrinkage.png', 'results/rtaac_shrinkage.pdf')
convert_png_to_pdf('results/rtaac_shrinkage.png', 'rtaac_shrinkage.pdf')
