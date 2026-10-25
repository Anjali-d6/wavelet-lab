# Algorithm

The wavelet method converts RGB images to luminance while leaving grayscale
images as-is. It applies a 2D discrete wavelet transform with symmetric boundary
extension, scales horizontal, vertical, and diagonal detail coefficients, makes
a restrained contrast adjustment to approximation coefficients (gain
`1 + 0.6 × strength`), and reconstructs
the image with the inverse transform. The output is cropped to the input size and
clipped to 8-bit range. Enhancement strength blends the requested detail scale
toward the identity transform.

Optional denoising estimates noise from diagonal detail coefficients using the
median absolute deviation and applies soft thresholding. This can remove fine
image structures as well as noise. CLAHE is applied to luminance using
scikit-image's adaptive histogram equalization. Gamma correction remaps
luminance using `output = input ** (1 / gamma)`.

The reported RMS contrast is the standard deviation of grayscale intensity.
PSNR and SSIM are only calculated against a user-provided reference with matching
dimensions and channel count. Neither brightness, contrast, PSNR, nor SSIM is a
measure of diagnostic accuracy.
