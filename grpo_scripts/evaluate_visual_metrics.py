
import os
import argparse
import numpy as np
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from scipy.ndimage import binary_dilation, gaussian_filter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_and_process(src):
    """Helper to load image and ensure RGB with White Background (handling RGBA transparency)"""
    try:
        img = Image.open(src)
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert('RGB')
    except Exception as e:
        print(f"[Error] Failed to load {src}: {e}")
        return None

def get_bbox(img_obj):
    """Detect content bounding box"""
    arr = np.array(img_obj)
    if len(arr.shape) == 2:
        mask = arr < 250
    else:
        mask = np.mean(arr, axis=2) < 250
    
    if not np.any(mask):
        return None
        
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (x_min, y_min, x_max, y_max)

def process_edge_map(img_arr):
    """Process image for edge detection"""
    # Convert to grayscale
    gray = 0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2]
    # Canny edge detection
    edges = canny(gray, sigma=1.0)
    # Dilation
    dilated = binary_dilation(edges, structure=np.ones((3,3)), iterations=1)
    # Gaussian Blur
    blurred = gaussian_filter(dilated.astype(float), sigma=2.0)
    return blurred

def main():
    parser = argparse.ArgumentParser(description="Evaluate visual similarity metrics between GT and Inference folders.")
    parser.add_argument("--gt_folder", type=str, default="/disk/CJN/ms-swift/grpo_test_beagle_gt", help="Path to Ground Truth folder")
    parser.add_argument("--infer_folder", type=str, default="/disk/CJN/Chart2SVG_dataset/test_data/grpo_test_beagle_ckp140_inference_results", help="Path to Inference result folder")
    args = parser.parse_args()

    gt_folder = args.gt_folder
    infer_folder = args.infer_folder

    if not os.path.exists(gt_folder):
        print(f"Error: GT folder '{gt_folder}' does not exist.")
        return
    if not os.path.exists(infer_folder):
        print(f"Error: Inference folder '{infer_folder}' does not exist.")
        return

    # Initialize LPIPS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    try:
        loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    except Exception as e:
        print(f"Warning: LPIPS failed to initialize: {e}")
        loss_fn = None

    # Get list of GT images
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_files.sort()

    total_pixel_score = 0.0
    total_edge_score = 0.0
    total_ssim_score = 0.0
    total_lpips_score = 0.0
    total_final_score = 0.0
    count = 0
    finals = []

    print(f"{'File':<40} | {'Pixel':<8} | {'Edge':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'Final':<8}")
    print("-" * 90)

    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        
        # Match with {file_name}_normalized.png in infer_folder
        file_root, _ = os.path.splitext(gt_file)
        infer_file_name = f"{file_root}_normalized.png"
        infer_path = os.path.join(infer_folder, infer_file_name)

        if not os.path.exists(infer_path):
            print(f"Warning: Corresponding inference file not found for {gt_file} (expected {infer_file_name})")
            continue

        gt_img = load_and_process(gt_path)
        gen_img = load_and_process(infer_path)

        if gt_img is None or gen_img is None:
            continue

        # 1. Ensure Gen Image matches GT Size
        if gen_img.size != gt_img.size:
            gen_img = gen_img.resize(gt_img.size, Image.Resampling.LANCZOS)

        # 2. Detect content bounding box and crop
        gt_bbox = get_bbox(gt_img)
        
        if gt_bbox:
            x_min, y_min, x_max, y_max = gt_bbox
            w, h = gt_img.size
            pad = 20 # Using 20 as per latest plugin.py logic
            
            crop_x_min = max(0, x_min - pad)
            crop_y_min = max(0, y_min - pad)
            crop_x_max = min(w, x_max + pad)
            crop_y_max = min(h, y_max + pad)
            
            crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
                
            gt_img_processed = gt_img.crop(crop_box)
            gen_img_processed = gen_img.crop(crop_box)
        else:
            gt_img_processed = gt_img
            gen_img_processed = gen_img

        # Convert to numpy
        gen_arr = np.array(gen_img_processed, dtype=np.float32)
        gt_arr = np.array(gt_img_processed, dtype=np.float32)

        # Check for Blank Gen Image
        is_blank = False
        if np.mean(gen_arr) > 254:
            is_blank = True

        # Edge Detection Pre-calculation
        gen_edge_map = None
        gt_edge_map = None
        no_edges = False

        if not is_blank:
            gen_edge_map = process_edge_map(gen_arr)
            gt_edge_map = process_edge_map(gt_arr)
            if np.sum(gen_edge_map) < 1e-6:
                no_edges = True

        if is_blank or no_edges:
            pixel_score_mapped = 0.0
            edge_score = 0.0
            ssim_score = 0.0
            lpips_score = 0.0
        else:
            # 1. Pixel-wise L2 Score
            gen_norm = gen_arr / 255.0
            gt_norm = gt_arr / 255.0
            pixel_mse = np.mean((gen_norm - gt_norm) ** 2)
            
            # 2. Edge Score
            edge_mse = np.mean((gen_edge_map - gt_edge_map) ** 2)
            
            # 3. SSIM
            try:
                ssim_val = ssim(gen_arr, gt_arr, channel_axis=2, data_range=255.0)
                ssim_score = max(0.0, ssim_val)
            except Exception:
                ssim_score = 0.0

            # 4. LPIPS
            lpips_score = 0.0
            if loss_fn:
                try:
                    t_gen = torch.from_numpy(gen_arr).float() / 127.5 - 1.0
                    t_gen = t_gen.permute(2, 0, 1).unsqueeze(0).to(device)
                    t_gt = torch.from_numpy(gt_arr).float() / 127.5 - 1.0
                    t_gt = t_gt.permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        dist = loss_fn(t_gen, t_gt)
                        lpips_val = dist.item()
                    lpips_score = max(0.0, 1.0 - lpips_val)
                except Exception:
                    lpips_score = ssim_score
            else:
                lpips_score = ssim_score

            # 5. Mapped Scores
            pixel_score_mapped = np.exp(-10.0 * pixel_mse)
            edge_score = np.exp(-10.0 * edge_mse)

        # Final Weighted Score (Weights from plugin.py)
        # raw_score = 0.4 * pixel_score_mapped + 0.15 * edge_score + 0.3 * ssim_score + 0.15 * lpips_score
        final_score = 0.4 * pixel_score_mapped + 0.15 * edge_score + 0.3 * ssim_score + 0.15 * lpips_score

        # print(f"{gt_file[:40]:<40} | {pixel_score_mapped:.4f}   | {edge_score:.4f}   | {ssim_score:.4f}   | {lpips_score:.4f}   | {final_score:.4f}")

        total_pixel_score += pixel_score_mapped
        total_edge_score += edge_score
        total_ssim_score += ssim_score
        total_lpips_score += lpips_score
        total_final_score += final_score
        count += 1
        finals.append(final_score)

    if count > 0:
        print("-" * 90)
        print(f"{'Average':<40} | {total_pixel_score/count:.4f}   | {total_edge_score/count:.4f}   | {total_ssim_score/count:.4f}   | {total_lpips_score/count:.4f}   | {total_final_score/count:.4f}")
        # Sort by final score descending
        finals_sorted = sorted(finals, reverse=True)
        def top_mean(pct):
            k = max(1, int(len(finals_sorted) * pct))
            return float(np.mean(finals_sorted[:k]))
        print(f"Top70% Avg Final: {top_mean(0.7):.4f}")
        print(f"Top80% Avg Final: {top_mean(0.8):.4f}")
        print(f"Top90% Avg Final: {top_mean(0.9):.4f}")
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    main()
