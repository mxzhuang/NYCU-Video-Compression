#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VC_HW4: Basic Image Compression Implementation

This script implements key steps of the JPEG compression pipeline:
8x8 DCT, Quantization, Zigzag Scan, and Run-Length Encoding (RLE),
along with the corresponding decoding process.

*** Change: This version uses a new, correct NumPy-based DCT/IDCT
    implementation. ***
"""

import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import math
import matplotlib.pyplot as plt

# --- Constants ---

# Quantization Table 1 (from spec)
# Provides better quality, lower compression
Q_TABLE_1 = np.array([
    [10, 7, 6, 10, 14, 24, 31, 37],
    [7, 7, 8, 11, 16, 35, 36, 33],
    [8, 8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
])

# Quantization Table 2 (from spec)
# Provides lower quality, higher compression
Q_TABLE_2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
])

# Zigzag scan indices (8x8)
ZIGZAG_INDICES = np.array([
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
])

# Inverse Zigzag scan indices
INVERSE_ZIGZAG_INDICES = np.argsort(ZIGZAG_INDICES)


# --- Pre-calculate DCT Matrix (Correct Implementation) ---

def create_dct_matrix(N=8):
    """
    Create the 1D DCT-II transformation matrix (C) using NumPy broadcasting.
    This is a vectorized implementation.
    """
    # Create arrays for n (columns) and k (rows)
    n = np.arange(N, dtype=np.float64)
    k = np.arange(N, dtype=np.float64).reshape(-1, 1) # (N, 1)
    
    # Calculate the cosine term
    # This broadcasts k (column vector) over n (row vector)
    cos_term = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    
    # Create scaling coefficients
    scales = np.ones(N) * np.sqrt(2.0 / N)
    scales[0] = np.sqrt(1.0 / N)
    
    # Apply scales row-wise (to each k)
    # This broadcasts scales (column vector) over the cos_term (N, N)
    matrix = cos_term * scales.reshape(-1, 1)
    
    return matrix

# Pre-calculate the 8x8 matrix (C) and its transpose (C.T)
DCT_MATRIX_8x8 = create_dct_matrix(8)
IDCT_MATRIX_8x8 = DCT_MATRIX_8x8.T

# --- Helper Functions ---

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    
    if mse == 0:
        return float('inf')
        
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


# --- Core Algorithm Functions ---

def dct2(block):
    """
    Apply 2D DCT using matrix multiplication.
    Y = C @ X @ C.T
    """
    return DCT_MATRIX_8x8 @ block @ IDCT_MATRIX_8x8

def idct2(block):
    """
    Apply 2D IDCT using matrix multiplication.
    X = C.T @ Y @ C
    """
    return IDCT_MATRIX_8x8 @ block @ DCT_MATRIX_8x8

def zigzag_scan(block):
    """
    Convert 8x8 block to 1x64 vector using zigzag order.
    """
    return block.flatten()[ZIGZAG_INDICES]

def inverse_zigzag_scan(vector):
    """
    Convert 1x64 vector back to 8x8 block.
    """
    if len(vector) != 64:
        raise ValueError("Vector length must be 64 for inverse zigzag scan")
    return vector.flatten()[INVERSE_ZIGZAG_INDICES].reshape((8, 8))

def run_length_encoding(vector):
    """
    Apply Run-Length Encoding (RLE) to the 1D zigzag vector.
    """
    encoded_data = []
    zero_count = 0
    for val in vector:
        if val == 0:
            zero_count += 1
        else:
            encoded_data.append((zero_count, int(val)))
            zero_count = 0
    encoded_data.append((0, 0)) # EOB marker
    return encoded_data

def run_length_decoding(rle_data):
    """
    Decode RLE data back into a 1x64 vector.
    """
    vector = np.zeros(64, dtype=int)
    index = 0
    for zero_count, value in rle_data:
        if (zero_count, value) == (0, 0): # EOB
            break
        index += zero_count
        if index < 64:
            vector[index] = value
            index += 1
        else:
            break
    return vector


# --- Main Encode/Decode Pipelines ---

def encode_image(image_array, q_table):
    """
    Run the full encoding pipeline on a grayscale image array.
    """
    height, width = image_array.shape
    encoded_blocks = []
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            # 1. Extract 8x8 block
            block = image_array[y:y+8, x:x+8]
            # 2. Level Shift: [0, 255] -> [-128, 127]
            block = block.astype(np.float32) - 128.0
            # 3. Apply 2D DCT
            dct_block = dct2(block)
            # 4. Quantize
            quantized_block = np.round(dct_block / q_table).astype(int)
            # 5. Zigzag Scan
            zigzag_vector = zigzag_scan(quantized_block)
            # 6. Run-Length Encoding
            rle_data = run_length_encoding(zigzag_vector)
            
            encoded_blocks.append(rle_data)
    return encoded_blocks

def decode_image(encoded_blocks, image_shape, q_table):
    """
    Run the full decoding pipeline to recover the image.
    """
    height, width = image_shape
    recovered_image_array = np.zeros(image_shape, dtype=np.float32)
    block_index = 0
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if block_index >= len(encoded_blocks):
                break
                
            rle_data = encoded_blocks[block_index]
            
            # 1. Run-Length Decoding
            zigzag_vector = run_length_decoding(rle_data)
            # 2. Inverse Zigzag Scan
            quantized_block = inverse_zigzag_scan(zigzag_vector)
            # 3. De-quantization
            dequantized_block = quantized_block.astype(np.float32) * q_table
            # 4. Apply 2D Inverse DCT
            idct_block = idct2(dequantized_block)          
            # 5. Reverse Level Shift: [-128, 127] -> [0, 255]
            recovered_block = idct_block + 128.0
            # 6. Place the 8x8 block back into the image
            recovered_image_array[y:y+8, x:x+8] = recovered_block
            
            block_index += 1
            
    recovered_image_array = np.clip(recovered_image_array, 0, 255)
    return recovered_image_array.astype(np.uint8)


# --- Main Execution ---

def main():
    """
    Main function to run the compression and decompression process.
    """
    
    # --- 1. Setup ---
    OUTPUT_DIR = "output_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BYTES_PER_SYMBOL = 2 # Estimate: 2 bytes per (zero_count, value) tuple
    
    # --- 2. Load Image ---
    try:
        img_pil_original = Image.open('lena.png')
    except FileNotFoundError:
        print("Error: 'lena.png' not found.")
        print("Please download 'lena.png' and place it in the same directory.")
        return
        
    img_pil_gray = img_pil_original.convert('L')
    original_shape = img_pil_gray.size[::-1] # (H, W) format

    # --- 3. Crop image to be a multiple of 8x8 ---
    width, height = img_pil_gray.size
    new_width = width - (width % 8)
    new_height = height - (height % 8)
    
    img_pil_cropped = img_pil_gray.crop((0, 0, new_width, new_height))
    img_array_cropped = np.array(img_pil_cropped, dtype=np.uint8) # NumPy array of original
    cropped_shape = img_array_cropped.shape
    
    # --- 4. Calculate Original Size (as pixel count) ---
    original_size_bytes = img_array_cropped.size # 8-bit grayscale = 1 pixel = 1 byte

    # --- 5. Process with Quantization Table 1 ---
    print("Processing with Quantization Table 1")
    print("============================================================")
    print(f"Original image shape: {original_shape}")
    print(f"Processed (cropped) shape: {cropped_shape}")
    
    encoded_data_1 = encode_image(img_array_cropped, Q_TABLE_1)
    
    encoded_symbols_1 = sum(len(block) for block in encoded_data_1)
    encoded_size_bytes_1 = encoded_symbols_1 * BYTES_PER_SYMBOL
    compression_ratio_1 = original_size_bytes / encoded_size_bytes_1
    
    print(f"Original size: {original_size_bytes} bytes")
    print(f"Encoded size (estimated): {encoded_size_bytes_1} bytes ({encoded_symbols_1} RLE symbols)")
    print(f"Compression ratio (estimated): {compression_ratio_1:.2f} : 1")
    

    # --- 6. Process with Quantization Table 2 ---
    print("\nProcessing with Quantization Table 2")
    print("============================================================")
    print(f"Original image shape: {original_shape}")
    print(f"Processed (cropped) shape: {cropped_shape}")
    
    encoded_data_2 = encode_image(img_array_cropped, Q_TABLE_2)
    
    encoded_symbols_2 = sum(len(block) for block in encoded_data_2)
    encoded_size_bytes_2 = encoded_symbols_2 * BYTES_PER_SYMBOL
    compression_ratio_2 = original_size_bytes / encoded_size_bytes_2

    print(f"Original size: {original_size_bytes} bytes")
    print(f"Encoded size (estimated): {encoded_size_bytes_2} bytes ({encoded_symbols_2} RLE symbols)")
    print(f"Compression ratio (estimated): {compression_ratio_2:.2f} : 1")


    # --- 7. Decoding images ---
    print("\nDecoding images")
    print("============================================================")
    
    # Decode Q1
    recovered_image_1_array = decode_image(encoded_data_1, cropped_shape, Q_TABLE_1)
    img_out_1_pil = Image.fromarray(recovered_image_1_array)
    save_path_1 = os.path.join(OUTPUT_DIR, "lena_recovered_table1.png")
    img_out_1_pil.save(save_path_1)
    print(f"Saved: {save_path_1}")
    
    # Decode Q2
    recovered_image_2_array = decode_image(encoded_data_2, cropped_shape, Q_TABLE_2)
    img_out_2_pil = Image.fromarray(recovered_image_2_array)
    save_path_2 = os.path.join(OUTPUT_DIR, "lena_recovered_table2.png")
    img_out_2_pil.save(save_path_2)
    print(f"Saved: {save_path_2}")
    
    # Create comparison plot with Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array_cropped, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original (Cropped)')
    axes[0].axis('off')
    
    axes[1].imshow(recovered_image_1_array, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Reconstructed (Q1)')
    axes[1].axis('off')
    
    axes[2].imshow(recovered_image_2_array, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Reconstructed (Q2)')
    axes[2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, 'comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    
    print(f"Comparison image saved: {comparison_path}")
    

    # --- 8. Image Quality Metrics ---
    print("\nImage Quality Metrics:")
    print("============================================================")
    
    psnr_1 = calculate_psnr(img_array_cropped, recovered_image_1_array)
    psnr_2 = calculate_psnr(img_array_cropped, recovered_image_2_array)
    
    print(f"PSNR with Q1: {psnr_1:.2f} dB")
    print(f"PSNR with Q2: {psnr_2:.2f} dB")

    # --- 9. Analysis ---
    print("\nAnalysis:")
    print("============================================================")
    print(f"Q1 uses finer quantization -> Better quality (Higher PSNR), larger size (Higher {BYTES_PER_SYMBOL}-Byte RLE symbols)")
    print(f"Q2 uses coarser quantization -> Lower quality (Lower PSNR), smaller size (Fewer {BYTES_PER_SYMBOL}-Byte RLE symbols)")
    print("============================================================")


if __name__ == "__main__":
    main()