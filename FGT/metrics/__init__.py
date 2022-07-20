import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_metrics(results, gts):
    B, H, W, C = results.shape
    psnr_values, ssim_values, L1errors, L2errors = [], [], [], []
    for i in range(B):
        result = results[i]
        gt = gts[i]
        result_img = result
        gt_img = gt
        residual = result - gt
        L1error = np.mean(np.abs(residual))
        L2error = np.sum(residual ** 2) ** 0.5 / (H * W * C)
        psnr_value = psnr(result_img, gt_img)
        ssim_value = ssim(result_img, gt_img, multichannel=True)
        L1errors.append(L1error)
        L2errors.append(L2error)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    L1_value = np.mean(L1errors)
    L2_value = np.mean(L2errors)
    psnr_value = np.mean(psnr_values)
    ssim_value = np.mean(ssim_values)

    return {'l1': L1_value, 'l2': L2_value, 'psnr': psnr_value, 'ssim': ssim_value}
