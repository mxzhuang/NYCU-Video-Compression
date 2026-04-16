import cv2
import numpy as np
import time
import os
import math
from matplotlib import pyplot as plt

# --- Helper Functions ---

def compute_psnr(img_original, img_reconstructed):
    """計算 PSNR."""
    img_original = img_original.astype(np.float64)
    img_reconstructed = img_reconstructed.astype(np.float64)
    
    mse = np.mean((img_original - img_reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # 影像完全相同
    
    max_pixel_value = 255.0
    psnr_db = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr_db

def save_output_images(reconstructed_img, residual_img, search_p, method_name, block_size):
    """儲存 reconstructed images 和 residual images."""
    output_dir = f'output/P_{search_p}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存 reconstructed images
    recon_filename = f'{output_dir}/recon_{method_name}_b{block_size}.png'
    cv2.imwrite(recon_filename, reconstructed_img)
    
    # 儲存 residual images
    residual_visual = cv2.normalize(residual_img, None, 0, 255, cv2.NORM_MINMAX)
    residual_filename = f'{output_dir}/residual_{method_name}_b{block_size}.png'
    cv2.imwrite(residual_filename, residual_visual)

# ---  Motion Compensation ---

def motion_compensation(ref_frame, curr_frame, motion_vectors, block_size):
    """
    根據運動向量，從參考幀重建當前幀
    """
    height, width = curr_frame.shape
    
    reconstructed_frame = np.zeros_like(curr_frame, dtype=np.uint8)
    residual_frame = np.zeros_like(curr_frame, dtype=np.float32)
    
    # 遍歷所有區塊索引
    for r_idx in range(height // block_size):
        for c_idx in range(width // block_size):
            
            # 獲取當前區塊的左上角座標
            r = r_idx * block_size
            c = c_idx * block_size
            
            # 獲取儲存的運動向量 (dx, dy)
            dx, dy = motion_vectors[r_idx, c_idx]
            
            # 計算在參考幀 (ref_frame) 中對應區塊的座標
            ref_r, ref_c = r + dy, c + dx
            
            # --- 進行運動補償 ---
            # 1. 取得預測區塊 (Predicted Block)
            predicted_block = ref_frame[ref_r : ref_r + block_size, 
                                        ref_c : ref_c + block_size]
            
            # 2. 放到重建幀的對應位置
            reconstructed_frame[r : r + block_size, 
                                c : c + block_size] = predicted_block
            
            # 3. 計算殘差 (Residual)
            current_block = curr_frame[r : r + block_size, 
                                       c : c + block_size]
            
            # 轉換為 float32 進行相減
            residual_block = current_block.astype(np.float32) - predicted_block.astype(np.float32)
            residual_frame[r : r + block_size, 
                           c : c + block_size] = residual_block

    return reconstructed_frame, residual_frame

# --- Motion Estimation ---

def motion_estimation_fs(ref_frame, curr_frame, block_size, search_p):
    """
    全域搜尋 (Full Search) 演算法。
    search_p: 搜尋範圍
    """
    start_time = time.time()
    height, width = curr_frame.shape
    
    # 建立一個陣列來儲存每個區塊的運動向量 (dx, dy)
    mv_rows = height // block_size
    mv_cols = width // block_size
    motion_vectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.int32)

    # 遍歷 curr_frame 中的每一個區塊
    for r_idx in range(mv_rows):
        for c_idx in range(mv_cols):
            
            # 當前區塊的左上角座標
            r = r_idx * block_size
            c = c_idx * block_size
            
            # 取得當前區塊
            current_block = curr_frame[r : r + block_size, 
                                       c : c + block_size].astype(np.float32)
            
            min_sad = float('inf')
            best_mv = (0, 0) # (dx, dy)

            # 在 ref_frame 的搜尋範圍內尋找最佳匹配
            # dy, dx 是運動向量的候選值
            for dy in range(-search_p, search_p + 1):
                for dx in range(-search_p, search_p + 1):
                    
                    # 計算候選區塊在 ref_frame 中的座標
                    ref_r, ref_c = r + dy, c + dx

                    # 邊界檢查 
                    if (ref_r < 0 or ref_r + block_size > height or
                        ref_c < 0 or ref_c + block_size > width):
                        continue 

                    # 取得候選區塊
                    ref_block = ref_frame[ref_r : ref_r + block_size, 
                                          ref_c : ref_c + block_size].astype(np.float32)
                    
                    # --- 計算 SAD ---
                    sad = np.sum(np.abs(current_block - ref_block))
                    
                    # --- 更新最佳匹配 ---
                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (dx, dy)
            
            # 儲存這個區塊的最佳運動向量
            motion_vectors[r_idx, c_idx] = best_mv
            
    end_time = time.time()
    runtime = end_time - start_time
    
    return motion_vectors, runtime


def motion_estimation_tss(ref_frame, curr_frame, block_size, search_p):
    """
    三步搜尋 (Three-Step Search, TSS) 演算法。
    """
    start_time = time.time()
    height, width = curr_frame.shape
    
    mv_rows = height // block_size
    mv_cols = width // block_size
    motion_vectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.int32)
    
    # 決定初始步長
    initial_step = 2**(int(math.log2(search_p))) // 2

    for r_idx in range(mv_rows):
        for c_idx in range(mv_cols):
            
            r = r_idx * block_size
            c = c_idx * block_size
            
            current_block = curr_frame[r : r + block_size, 
                                       c : c + block_size].astype(np.float32)
            
            # 搜尋的中心點，初始為 (r, c)
            center_r, center_c = r, c
            
            current_step = initial_step
            
            # TSS 迴圈，直到步長為 1
            while current_step >= 1:
                
                search_points = [(0, 0), (0, current_step), (0, -current_step), 
                                 (current_step, 0), (-current_step, 0),
                                 (current_step, current_step), (current_step, -current_step), 
                                 (-current_step, current_step), (-current_step, -current_step)]
                
                min_sad_step = float('inf')
                best_center_pos = (center_r, center_c) # 儲存這一步的最佳中心

                # 檢查 9 個點
                for dr, dc in search_points:
                    # 計算候選點在 ref_frame 中的實際座標
                    ref_r, ref_c = center_r + dr, center_c + dc
                    
                    # 邊界檢查
                    if (ref_r < 0 or ref_r + block_size > height or
                        ref_c < 0 or ref_c + block_size > width):
                        continue

                    ref_block = ref_frame[ref_r : ref_r + block_size, 
                                          ref_c : ref_c + block_size].astype(np.float32)
                    
                    sad = np.sum(np.abs(current_block - ref_block))
                    
                    if sad < min_sad_step:
                        min_sad_step = sad
                        best_center_pos = (ref_r, ref_c)
                
                # 更新中心點
                center_r, center_c = best_center_pos
                
                # 步長減半 
                current_step //= 2

            # 迴圈結束 (step < 1)，最終的 center_r, center_c 就是最佳位置
            best_dx = center_c - c
            best_dy = center_r - r
            motion_vectors[r_idx, c_idx] = (best_dx, best_dy)

    end_time = time.time()
    runtime = end_time - start_time
    
    return motion_vectors, runtime

# --- 實驗與繪圖 ---

def run_all_experiments(ref_path, curr_path, block_size, search_ranges_list):
    """
    執行作業中的所有比較實驗。
    """
    try:
        ref_frame = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        curr_frame = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
        if ref_frame is None or curr_frame is None:
            print(f"錯誤：無法讀取影像。請檢查路徑：{ref_path}, {curr_path}")
            return
    except Exception as e:
        print(f"讀取影像時發生錯誤: {e}")
        return

    print(f"開始實驗... 區塊大小: {block_size}x{block_size}")
    
    # 儲存所有結果
    results = {
        'search_ranges': search_ranges_list,
        'FS_psnr': [], 'FS_time': [],
        'TSS_psnr': [], 'TSS_time': []
    }
    
    for p in search_ranges_list:
        print(f"\n--- 測試搜尋範圍: [+/-{p}] ---")
        
        # --- 1. 全域搜尋 (Full Search) ---
        mv_fs, time_fs = motion_estimation_fs(ref_frame, curr_frame, block_size, p)
        recon_fs, residual_fs = motion_compensation(ref_frame, curr_frame, mv_fs, block_size)
        psnr_fs = compute_psnr(curr_frame, recon_fs)
        
        # 儲存 FS 結果
        results['FS_psnr'].append(psnr_fs)
        results['FS_time'].append(time_fs)
        save_output_images(recon_fs, residual_fs, p, 'FS', block_size)
        print(f"  Full Search (FS):    PSNR: {psnr_fs:7.3f} dB, Runtime: {time_fs:7.3f} sec")

        # --- 2. 三步搜尋 (Three-Step Search) ---
        mv_tss, time_tss = motion_estimation_tss(ref_frame, curr_frame, block_size, p)
        recon_tss, residual_tss = motion_compensation(ref_frame, curr_frame, mv_tss, block_size)
        psnr_tss = compute_psnr(curr_frame, recon_tss)

        # 儲存 TSS 結果
        results['TSS_psnr'].append(psnr_tss)
        results['TSS_time'].append(time_tss)
        save_output_images(recon_tss, residual_tss, p, 'TSS', block_size)
        print(f"  Three-Step (TSS):  PSNR: {psnr_tss:7.3f} dB, Runtime: {time_tss:7.3f} sec")
        
        # --- 3. 比較 ---
        print(f"  > FS vs TSS (PSNR):  {psnr_fs - psnr_tss:+.3f} dB (FS 較佳)")
        print(f"  > FS vs TSS (Time):  TSS 節省 {time_fs - time_tss:.3f} sec")

    return results

def plot_comparison_charts(results_data):
    """
    根據實驗結果繪製 PSNR 和 Runtime 的比較圖。
    """
    plot_dir = 'output/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    search_ranges = results_data['search_ranges']
    
    # --- 繪製 PSNR 圖 ---
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(search_ranges, results_data['FS_psnr'], 'o-', label='Full Search (FS)')
    plt.plot(search_ranges, results_data['TSS_psnr'], 's-', label='Three-Step Search (TSS)')
    plt.title('PSNR vs. Search Range')
    plt.xlabel('Search Range (+/- p)')
    plt.ylabel('PSNR (dB)')
    plt.xticks(search_ranges)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- 繪製 Runtime 圖 ---
    plt.subplot(1, 2, 2)
    plt.plot(search_ranges, results_data['FS_time'], 'o-', label='Full Search (FS)')
    plt.plot(search_ranges, results_data['TSS_time'], 's-', label='Three-Step Search (TSS)')
    plt.title('Runtime vs. Search Range')
    plt.xlabel('Search Range (+/- p)')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(search_ranges)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 儲存圖表
    plot_filename = f'{plot_dir}/comparison_chart.png'
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"\n比較圖表已儲存至: {plot_filename}")

# --- 5. 主程式執行區 ---

if __name__ == "__main__":
    
    # --- 作業設定 ---
    REF_IMAGE_PATH = 'one_gray.png'
    CURR_IMAGE_PATH = 'two_gray.png'
    BLOCK_SIZE = 8
    SEARCH_RANGES = [8, 16, 32] 
    
    # 建立主輸出資料夾
    os.makedirs('output', exist_ok=True)
    
    # 執行所有實驗
    all_results = run_all_experiments(
        REF_IMAGE_PATH, 
        CURR_IMAGE_PATH, 
        BLOCK_SIZE, 
        SEARCH_RANGES
    )
    
    # 繪製並儲存結果圖表
    if all_results:
        plot_comparison_charts(all_results)
        print("\n實驗完成！所有影像和圖表已儲存在 'output' 資料夾中。")