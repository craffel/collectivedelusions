with open('submission/sections/03_method.tex', 'r') as f:
    text = f.read()

# Make Minor Suggestion 2 fixes:
text = text.replace(
    r"\Lambda \in [0, 1]^{L \times K}",
    r"$\Lambda \in [0, 1]^{L \times K}$"
)
text = text.replace(
    r"L \times K = 56",
    r"$L \times K = 56$"
)
text = text.replace(
    "initialized at 0.3,",
    "initialized at $0.3$,"
)

# Make Minor Suggestion 3 fixes (convolutional weights):
old_conv = r"For any weight tensor at layer $l$, scale factors are computed dynamically along each output channel $c$. Multi-dimensional tensors are flattened to 2D matrices, with scale factors computed along output channels."
new_conv = r"For any weight tensor at layer $l$, scale factors are computed dynamically along each output channel $c$. In our Vision Transformer backbone, projection layers (such as the patch embedding block) utilize 4D convolutional weight tensors of shape $[C_{\text{out}}, C_{\text{in}}, H_{\text{kernel}}, W_{\text{kernel}}]$. To perform per-channel quantization on these multi-dimensional tensors, we directly compute scale factors along the output channel axis ($C_{\text{out}}$) without flattening, ensuring that each 3D kernel slice $[c, :, :, :]$ is scaled independently. For standard 2D weight matrices, scale factors are computed along the out-features dimension directly."

if old_conv in text:
    text = text.replace(old_conv, new_conv)
    print("Successfully replaced convolutional description.")
else:
    print("Warning: Convolutional description mismatch.")

with open('submission/sections/03_method.tex', 'w') as f:
    f.write(text)

print("Done with 03_method.tex editing.")
