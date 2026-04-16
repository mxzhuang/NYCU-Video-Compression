# Required libraries: pip install numpy matplotlib Pillow opencv-python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import cv2

# Mathematical Helper Function
def C(k):
    """
    Calculates the C(k) scaling factor used in DCT formulas.
    Returns 1/sqrt(2) for k=0, and 1 otherwise.
    """
    return 1 / math.sqrt(2) if k == 0 else 1

def dct_2d(f_xy):
    """
    Computes the 2D-DCT based on its formula.
    """
    M, N = f_xy.shape
    F_uv = np.zeros_like(f_xy, dtype=np.float64)
    
    x = np.arange(M).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)

    print(f"Calculating 2D-DCT ...")
    for u in range(M):
        for v in range(N):
            cos_u = np.cos((2 * x + 1) * u * math.pi / (2 * M))
            cos_v = np.cos((2 * y + 1) * v * math.pi / (2 * N))
            summation = np.sum(f_xy * cos_u * cos_v)
            scale_factor = (2 / N) * C(u) * C(v)
            F_uv[u, v] = scale_factor * summation
            
    return F_uv

def idct_2d(F_uv):
    """
    Computes the 2D-IDCT, corresponding to the idct_2d formulas.
    """
    M, N = F_uv.shape
    f_xy = np.zeros_like(F_uv, dtype=np.float64)
    
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)

    c_u = np.full(M, 1.0); c_u[0] = 1 / math.sqrt(2)
    c_v = np.full(N, 1.0); c_v[0] = 1 / math.sqrt(2)
    scale_matrix = np.outer(c_u, c_v)

    print(f"Calculating 2D-IDCT ...")
    for x in range(M):
        for y in range(N):
            cos_u = np.cos((2 * x + 1) * u * math.pi / (2 * M))
            cos_v = np.cos((2 * y + 1) * v * math.pi / (2 * N))
            summation = np.sum(scale_matrix * F_uv * cos_u * cos_v)
            f_xy[x, y] = (2 / N) * summation
            
    return f_xy

def dct_two_1d(f_xy):
    """
    Computes the 2D-DCT by applying 1D-DCT function twice,
    """
    def dct_1d(f_x):
        N = len(f_x)
        F_u = np.zeros(N)
        x = np.arange(N)
        
        for u in range(N):
            cos_terms = np.cos((2 * x + 1) * u * math.pi / (2 * N))
            summation = f_x @ cos_terms
            scale_factor = math.sqrt(2 / N) * C(u)
            F_u[u] = scale_factor * summation
        return F_u

    print(f"Calculating Two 1D-DCT ...")
    intermediate = np.apply_along_axis(dct_1d, 1, f_xy)
    F_uv = np.apply_along_axis(dct_1d, 0, intermediate)
    
    return F_uv

def idct_two_1d(F_uv):
    """
    Computes the 2D-IDCT, corresponding to the dct_two_1d function.
    """
    def idct_1d(F_u):
        N = len(F_u)
        f_x = np.zeros(N)
        u = np.arange(N)
        
        c_u = np.full(N, 1.0); c_u[0] = 1 / math.sqrt(2)
        
        for x in range(N):
            cos_terms = np.cos((2 * x + 1) * u * math.pi / (2 * N))
            summation = (c_u * F_u) @ cos_terms
            scale_factor = math.sqrt(2 / N)
            f_x[x] = scale_factor * summation
        return f_x

    intermediate = np.apply_along_axis(idct_1d, 1, F_uv)
    f_xy = np.apply_along_axis(idct_1d, 0, intermediate)
    
    return f_xy


#  Helper Functions 
def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_visual_dct_image(dct_coeffs):
    """Generates a visually enhanced 8-bit image from DCT coefficients for display."""
    log_coeffs = np.log1p(np.abs(dct_coeffs))
    ac_coeffs = log_coeffs[1:, 1:]
    vmin = np.min(ac_coeffs)
    vmax = np.percentile(ac_coeffs, 99)
    clipped_coeffs = np.clip(log_coeffs, vmin, vmax)
    scale = 255.0 / (vmax - vmin)
    visual_image = ((clipped_coeffs - vmin) * scale).astype(np.uint8)
    return visual_image


#  Function for Reporting and Visualization
def save_and_visualize(all_results, output_folder):
    """
    Prints a comparison report, saves all result images, and displays a final comparison plot.
    """
    print("\n" + "="*25 + " Final Comparison Report " + "="*25)
    
    # Unpack results for clarity
    label_2d, F_uv_2d, recon_2d, time_2d, psnr_2d = all_results[0]
    label_1d, F_uv_1d, recon_1d, time_1d, psnr_1d = all_results[1]
    label_cv2, F_uv_cv2, recon_cv2, time_cv2, psnr_cv2 = all_results[2]
    
    # Print text report
    print(f"Method 1 ({label_2d})      Execution time: {time_2d:.6f} sec, PSNR: {psnr_2d:.2f} dB")
    print(f"Method 2 ({label_1d})    Execution time: {time_1d:.6f} sec, PSNR: {psnr_1d:.2f} dB")
    print(f"Method 3 ({label_cv2})      Execution time: {time_cv2:.6f} sec, PSNR: {psnr_cv2:.2f} dB")
    print("=" * 70)
    
    # Prepare images for saving and display
    visual_dct_2d = get_visual_dct_image(F_uv_2d)
    visual_dct_1d = get_visual_dct_image(F_uv_1d)
    visual_dct_cv2 = get_visual_dct_image(F_uv_cv2)
    
    recon_img_2d_8u = np.clip(recon_2d, 0, 255).astype(np.uint8)
    recon_img_1d_8u = np.clip(recon_1d, 0, 255).astype(np.uint8)
    recon_img_cv2_8u = np.clip(recon_cv2, 0, 255).astype(np.uint8)

    # Save images to disk
    plt.imsave(os.path.join(output_folder, "dct_coeffs_2d.png"), visual_dct_2d, cmap='gray')
    plt.imsave(os.path.join(output_folder, "reconstructed_2d.png"), recon_img_2d_8u, cmap='gray')
    plt.imsave(os.path.join(output_folder, "dct_coeffs_two1d.png"), visual_dct_1d, cmap='gray')
    plt.imsave(os.path.join(output_folder, "reconstructed_two1d.png"), recon_img_1d_8u, cmap='gray')
    plt.imsave(os.path.join(output_folder, "dct_coeffs_cv2.png"), visual_dct_cv2, cmap='gray')
    plt.imsave(os.path.join(output_folder, "reconstructed_cv2.png"), recon_img_cv2_8u, cmap='gray')
    
    # Display the final comparison plot
    print("\nAll computations and savings are complete. Displaying comparison plot...")
    N = recon_2d.shape[0] # Get image size for text placement
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('DCT Implementations Comparison & Validation', fontsize=20)
    axes[0, 0].set_title("DCT Coefficients", fontsize=14)
    axes[0, 1].set_title("Reconstructed Image", fontsize=14)
    
    plot_data = [
        (label_2d, visual_dct_2d, recon_img_2d_8u, time_2d, psnr_2d),
        (label_1d, visual_dct_1d, recon_img_1d_8u, time_1d, psnr_1d),
        (label_cv2, visual_dct_cv2, recon_img_cv2_8u, time_cv2, psnr_cv2)
    ]
    
    for i, (label, dct_img, recon_img, t, p) in enumerate(plot_data):
        axes[i, 0].set_ylabel(label, fontsize=14, fontweight='bold', labelpad=20)
        axes[i, 0].imshow(dct_img, cmap='gray'); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].text(10, N - 15, f"Time: {t:.4f}s", color='lime', backgroundcolor='black', fontsize=12)
        axes[i, 1].imshow(recon_img, cmap='gray'); axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].text(10, N - 15, f"PSNR: {p:.2f} dB", color='lime', backgroundcolor='black', fontsize=12)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Main Execution Block ---
def main():
    image_path = "lena.png"
    output_folder = "output_images"
    if not os.path.exists(output_folder): os.makedirs(output_folder)
        
    try:
        with Image.open(image_path) as img:
            original_img = np.array(img.convert('L'))
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return
        
    f_xy = np.float64(original_img)
    print(f"Successfully loaded and converted image to grayscale. Size: {f_xy.shape[0]}x{f_xy.shape[1]}")
    plt.imsave(os.path.join(output_folder, "original_grayscale.png"), original_img, cmap='gray')
    print("-" * 30)
    
    # Method 1: 2D-DCT 
    print("\nExecuting Method 1: 2D-DCT ...")
    start_time_2d = time.time()
    F_uv_2d = dct_2d(f_xy)
    time_2d = time.time() - start_time_2d
    reconstructed_f_xy_2d = idct_2d(F_uv_2d)
    psnr_2d = calculate_psnr(original_img, reconstructed_f_xy_2d)
    
    # Method 2: Two 1D-DCT 
    print("\nExecuting Method 2: Two 1D-DCT ...")
    start_time_1d = time.time()
    F_uv_1d = dct_two_1d(f_xy)
    time_1d = time.time() - start_time_1d
    reconstructed_f_xy_1d = idct_two_1d(F_uv_1d)
    psnr_1d = calculate_psnr(original_img, reconstructed_f_xy_1d)
    
    # Method 3: OpenCV built-in functions for validation 
    print("\nExecuting Method 3: OpenCV (for validation)...")
    start_time_cv2 = time.time()
    F_uv_cv2 = cv2.dct(f_xy.astype(np.float32))
    time_cv2 = time.time() - start_time_cv2
    reconstructed_f_xy_cv2 = cv2.idct(F_uv_cv2)
    psnr_cv2 = calculate_psnr(original_img, reconstructed_f_xy_cv2)
    
    # Collate all results into a single list
    all_results = [
        ("My 2D-DCT", F_uv_2d, reconstructed_f_xy_2d, time_2d, psnr_2d),
        ("My Two 1D-DCT", F_uv_1d, reconstructed_f_xy_1d, time_1d, psnr_1d),
        ("OpenCV Validation", F_uv_cv2, reconstructed_f_xy_cv2, time_cv2, psnr_cv2)
    ]
    
    # Call the function for saving images and visualization
    save_and_visualize(all_results, output_folder)

if __name__ == "__main__":
    main()