import asyncio
import os
import random
import re
import textwrap
import cairosvg
from lxml import etree
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from scipy.ndimage import binary_dilation, gaussian_filter, center_of_mass

# Ensure Chart2SVG is in path for importing semantic_tokens
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from Chart2SVG.data.semantic_tokens import svg2syntactic, syntactic2svg


from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# from swift.plugin import AsyncORM
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""

class SVGSyntaxReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        import cairosvg
        
        for completion in completions:
            # Simple check for svg tags
            if '<svg' in completion and '</svg>' in completion:
                try:
                    start = completion.find('<svg')
                    end = completion.rfind('</svg>') + 6
                    svg_code = completion[start:end]
                    
                    # Try rendering to verify syntax/renderability
                    # If cairosvg can convert it to png without error, it's syntactically valid enough
                    cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
                    rewards.append(1.0)
                except Exception:
                    rewards.append(-1.0)
            else:
                rewards.append(0.0)
        return rewards


class SVGStructureReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        images = kwargs.get('images', [])
        rewards = []
        # If images is not provided, try to find other sources or return 0
        if not images:
            return [0.0] * len(completions)

        for idx, completion in enumerate(completions):
            try:
                # Get Knowledge JSON path
                # Handle batch logic: images might be [batch_size] or [group_size]
                # If images list is shorter than completions, cycle or use 0?
                target_image = images[idx] if idx < len(images) else images[0]
                if isinstance(target_image, list):
                    target_image = target_image[0]
                
                img_path = target_image
                
                # Construct JSON path
                # /path/to/image.png -> /path/to/image.json
                json_path = os.path.splitext(img_path)[0] + '.json'
                
                if not os.path.exists(json_path):
                    rewards.append(0.0)
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)

                # Extract expected texts
                expected_texts = set()
                # 1. Title & Labels
                spec = knowledge.get('chart_specification', {})
                if spec.get('title'): expected_texts.add(str(spec['title']))
                if spec.get('x_label'): expected_texts.add(str(spec['x_label']))
                if spec.get('y_label'): expected_texts.add(str(spec['y_label']))
                
                # 2. Data
                chart_data = knowledge.get('chart_data', [])
                for item in chart_data:
                    for v in item.values():
                        expected_texts.add(str(v))
                
                # Check presence in SVG
                from lxml import etree
                if '<svg' not in completion or '</svg>' not in completion:
                    rewards.append(0.0)
                    continue
                    
                start = completion.find('<svg')
                end = completion.rfind('</svg>') + 6
                svg_code = completion[start:end]
                
                try:
                    root = etree.fromstring(svg_code.encode('utf-8'))
                    # Extract all text from SVG
                    svg_text_content = " ".join([elem.text for elem in root.iter() if elem.text])
                except:
                    # If parsing fails, use raw string search
                    svg_text_content = svg_code
                
                hit_count = 0
                total_count = len(expected_texts)
                if total_count == 0:
                    rewards.append(1.0)
                    continue
                    
                for text in expected_texts:
                    # Simple substring check
                    if str(text) in svg_text_content:
                        hit_count += 1
                
                rewards.append(hit_count / total_count)
                
            except Exception:
                rewards.append(0.0)
        return rewards



orms['svg_syntax'] = SVGSyntaxReward
orms['svg_structure'] = SVGStructureReward


class SVGSemanticVisualReward(ORM):
    """
    Reward = Semantic Executability + Visual Similarity
    Curriculum-style:
      - Semantic is a gate
      - Visual only contributes if semantic is complete
    """

    def __init__(
        self,
        semantic_weight: float = 0.3,
        visual_weight: float = 0.7,
        render_size=(512, 512)
    ):
        self.semantic_weight = semantic_weight
        self.visual_weight = visual_weight
        self.render_size = render_size
        self.last_semantic_scores = []
        self.last_visual_scores = []
        
        # Initialize LPIPS on CPU to save GPU memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # AlexNet is faster and standard for LPIPS metrics
            self.loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        except Exception as e:
            print(f"[RW Plugin] Warning: LPIPS failed to initialize: {e}")
            self.loss_fn = None

    @staticmethod
    def _extract_svg(completion: str) -> str:
        # If standard SVG tags are present, extract them to avoid trailing text issues
        if '<svg' in completion and '</svg>' in completion:
            start = completion.find('<svg')
            end = completion.rfind('</svg>') + 6
            return completion[start:end]
        
        # Otherwise return stripped content (assuming semantic tokens or partial generation)
        return completion.strip()

    def __call__(self, completions, **kwargs) -> List[float]:
        images = kwargs.get('images', [])
        rewards = []

        def _is_valid_color(val: str) -> bool:
            v = val.strip()
            if v.lower() == 'none':
                return True
            if re.fullmatch(r'#[0-9a-fA-F]{3}', v) or re.fullmatch(r'#[0-9a-fA-F]{6}', v):
                return True
            if re.fullmatch(r'rgb\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)', v):
                nums = list(map(int, re.findall(r'\d{1,3}', v)))
                return all(0 <= c <= 255 for c in nums[:3])
            if re.fullmatch(r'[a-zA-Z]+', v):
                return True
            return False

        def _sanitize_colors(xml_code: str) -> str:
            def repl_attr(m):
                attr = m.group(1)
                val = m.group(2)
                return f'{attr}="{val}"' if _is_valid_color(val) else f'{attr}="#000000"'
            xml_code = re.sub(r'(fill|stroke|stop-color|color)\s*=\s*"([^"]+)"', repl_attr, xml_code, flags=re.IGNORECASE)
            
            def repl_style(m):
                style_content = m.group(1)
                parts = style_content.split(';')
                new_parts = []
                for part in parts:
                    if ':' in part:
                        prop, val = part.split(':', 1)
                        prop = prop.strip().lower()
                        val = val.strip()
                        if prop in ['fill', 'stroke', 'stop-color', 'color']:
                            if not _is_valid_color(val):
                                val = '#000000'
                        new_parts.append(f"{prop}:{val}")
                    else:
                        new_parts.append(part)
                return f'style="{";".join(new_parts)}"'
            
            xml_code = re.sub(r'style="([^"]+)"', repl_style, xml_code, flags=re.IGNORECASE)
            xml_code = re.sub(r'#(?=[^0-9a-fA-F])', '#000000', xml_code)
            return xml_code

        for idx, completion in enumerate(completions):
            # ---------- 1. Semantic Executability ----------
            svg_code = self._extract_svg(completion)
            semantic_score = 0.0
            gen_img = None

            if svg_code:
                semantic_score = 0.2  # has svg-like structure

                if '[<|END_OF_SVG|>]' in svg_code:
                    semantic_score = 0.4

                try:
                    if '[<|START_OF_SVG|>]' in svg_code:
                        xml_code = syntactic2svg(svg_code)
                    else:
                        xml_code = svg_code

                    # Ensure xml_code contains only one SVG block (the first one) to avoid "Extra content" error
                    if '<svg' in xml_code and '</svg>' in xml_code:
                        start = xml_code.find('<svg')
                        end = xml_code.find('</svg>', start)
                        if start != -1 and end != -1:
                            xml_code = xml_code[start:end+6]

                    # XML parse success
                    etree.fromstring(xml_code.encode('utf-8'))
                    semantic_score = 0.6

                    # Try rendering
                    try:
                        png_data = cairosvg.svg2png(bytestring=xml_code.encode('utf-8'))
                    except Exception as e:
                        try:
                            xml_code = _sanitize_colors(xml_code)
                            png_data = cairosvg.svg2png(bytestring=xml_code.encode('utf-8'))
                        except Exception as e2:
                            # If rendering fails again, print the error and set gen_img to None
                            # so we can still get semantic score up to 0.6 but no visual score
                            print(f"[SVG Reward Render Error] {str(e2)}")
                            png_data = None
                            
                    if png_data:
                        gen_img = Image.open(io.BytesIO(png_data)).convert('RGB')
                        semantic_score = 1.0
                    else:
                        gen_img = None
                        semantic_score = 0.6 # Keep at 0.6 if XML valid but render failed

                except Exception as e:
                    print(f"[SVG Reward Error] {str(e)}")
                    import traceback
                    traceback.print_exc()
                    pass

            # ---------- 2. Visual Similarity ----------
            visual_score = 0.0
            pixel_score_mapped = 0.0
            edge_score = 0.0
            ssim_score = 0.0
            lpips_score = 0.0
            
            if semantic_score == 1.0 and gen_img is not None:
                try:
                    # Determine GT path logic for GRPO (group generation)
                    # User case: idx 0-3 -> img0, idx 4-7 -> img1 (Batch Size=2, Group Size=4)
                    gt_path = None
                    num_images = len(images) if images else 0
                    
                    if num_images > 0:
                        if num_images == 1:
                            gt_path = images[0] # Broadcast single GT to all completions
                        else:
                            # Map completion index to image index
                            # Assuming equal number of generations per image (standard GRPO)
                            generations_per_image = len(completions) // num_images
                            if generations_per_image > 0:
                                img_idx = idx // generations_per_image
                                if img_idx < num_images:
                                    gt_path = images[img_idx]
                                    print(f"[SVG Debug] idx={idx}, num_imgs={num_images}, mapped_to_img_idx={img_idx}")
                    
                    if gt_path:
                        if isinstance(gt_path, list):
                            gt_path = gt_path[0]
                        
                        gt_img = None
                        
                        # Helper to load image and ensure RGB with White Background (handling RGBA transparency)
                        def load_and_process(src):
                            try:
                                img = src if hasattr(src, 'mode') else Image.open(src)
                                if img.mode == 'RGBA':
                                    bg = Image.new('RGB', img.size, (255, 255, 255))
                                    bg.paste(img, mask=img.split()[3])
                                    return bg
                                return img.convert('RGB')
                            except Exception as e:
                                print(f"[SVG Load Error] {e}")
                                return None

                        # Handle dictionary (e.g. from datasets)
                        if isinstance(gt_path, dict):
                            if 'bytes' in gt_path:
                                # io is already imported at top level
                                img_bytes = gt_path.get('bytes')
                                if img_bytes:
                                    bio = io.BytesIO(img_bytes)
                                    bio.seek(0)
                                    gt_img = load_and_process(bio)
                                else:
                                    # Fallback: check if path exists in the dict and try to load from path
                                    if 'path' in gt_path and gt_path['path'] and os.path.exists(gt_path['path']):
                                         gt_img = load_and_process(gt_path['path'])
                            elif 'path' in gt_path:
                                gt_img = load_and_process(gt_path['path'])
                        
                        # Handle PIL Image object
                        elif hasattr(gt_path, 'convert'): 
                            gt_img = load_and_process(gt_path)

                        # Handle file path
                        elif gt_img is None and isinstance(gt_path, (str, bytes, os.PathLike)) and os.path.exists(gt_path):
                            gt_img = load_and_process(gt_path)

                        if gt_img is not None:
                            # --- Enhanced Visual Scoring Strategy with Auto-Crop ---
                            
                            def get_bbox(img_obj):
                                # Convert to numpy for fast processing
                                arr = np.array(img_obj)
                                # Check if grayscale or RGB
                                if len(arr.shape) == 2:
                                    mask = arr < 250
                                else:
                                    mask = np.mean(arr, axis=2) < 250
                                
                                if not np.any(mask):
                                    return (0, 0, img_obj.size[0], img_obj.size[1])
                                    
                                coords = np.argwhere(mask)
                                if coords.size == 0:
                                    return (0, 0, img_obj.size[0], img_obj.size[1])

                                y_min, x_min = coords.min(axis=0)
                                y_max, x_max = coords.max(axis=0)
                                return (x_min, y_min, x_max, y_max)

                            # 1. Ensure Gen Image matches GT Size (Pre-requisite for shared crop)
                            if gen_img.size != gt_img.size:
                                gen_img = gen_img.resize(gt_img.size, Image.Resampling.LANCZOS)

                            # 2. Detect content bounding box from GT (Reference)
                            gt_bbox = get_bbox(gt_img)
                            
                            if gt_bbox:
                                print(f"[SVG Debug] GT BBox detected: {gt_bbox}")
                                # Strict Crop Strategy: Crop to bounding box + padding
                                # Remove all white margins (top, bottom, left, right)
                                x_min, y_min, x_max, y_max = gt_bbox
                                w, h = gt_img.size
                                pad = 0 # Padding to avoid cutting off labels
                                
                                crop_x_min = max(0, x_min - pad)
                                crop_y_min = max(0, y_min - pad)
                                crop_x_max = min(w, x_max + pad)
                                crop_y_max = min(h, y_max + pad)
                                
                                crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
                                    
                                gt_img_processed = gt_img.crop(crop_box)
                                gen_img_processed = gen_img.crop(crop_box)
                            else:
                                # Fallback if GT is blank
                                gt_img_processed = gt_img
                                gen_img_processed = gen_img

                            # 3. No Resize - Use cropped images directly
                            # Convert to numpy (dimensions will match because same crop applied)
                            gen_arr = np.array(gen_img_processed, dtype=np.float32)
                            gt_arr = np.array(gt_img_processed, dtype=np.float32)

                            # Check for Blank Gen Image
                            # We already checked for blank during bbox detection but we need to check the processed array
                            is_blank = False
                            if np.mean(gen_arr) > 254: # Almost all white
                                is_blank = True
                            
                            def process_edge_map(img_arr):
                                # Convert to grayscale
                                gray = 0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2]
                                # Canny edge detection (sigma=1.0 is standard)
                                edges = canny(gray, sigma=1.0)
                                # Dilation (3x3 kernel, 1 iteration)
                                dilated = binary_dilation(edges, structure=np.ones((3,3)), iterations=1)
                                # Gaussian Blur
                                blurred = gaussian_filter(dilated.astype(float), sigma=2.0)
                                return blurred

                            # Prepare for checks
                            no_edges = False
                            gen_edge_map = None
                            gt_edge_map = None
                            
                            if not is_blank:
                                # B. Edge Detection
                                gen_edge_map = process_edge_map(gen_arr)
                                gt_edge_map = process_edge_map(gt_arr)
                                
                                # Check if no edges detected
                                if np.sum(gen_edge_map) < 1e-6:
                                    no_edges = True

                            # Combined Check: Blank or No-Edge
                            if is_blank or no_edges:
                                print(f"[SVG Debug] Blank or No-Edge detected for idx {idx}")
                                visual_score = 0.0
                                # Set components to 0.0
                                pixel_score_mapped = 0.0
                                edge_score = 0.0
                                ssim_score = 0.0
                                lpips_score = 0.0
                            else:
                                # 1. Pixel-wise L2 Score (Unnormalized, Linear Scale)
                                gen_norm = gen_arr / 255.0
                                gt_norm = gt_arr / 255.0
                                
                                # Calculate basic pixel MSE
                                pixel_mse = np.mean((gen_norm - gt_norm) ** 2)
                                
                                # 2. Edge Score (Unnormalized)
                                # Directly calculate MSE between edge maps
                                edge_mse = np.mean((gen_edge_map - gt_edge_map) ** 2)
                                
                                # 3. SSIM (Structural Similarity)
                                try:
                                    ssim_val = ssim(gen_arr, gt_arr, channel_axis=2, data_range=255.0)
                                    ssim_score = max(0.0, ssim_val)
                                except Exception as e:
                                    print(f"[RW Plugin] SSIM Error: {e}")
                                    ssim_score = 0.0

                                # 4. LPIPS
                                lpips_score = 0.0
                                if self.loss_fn:
                                    try:
                                        t_gen = torch.from_numpy(gen_arr).float().clone() / 127.5 - 1.0
                                        t_gen = t_gen.permute(2, 0, 1).unsqueeze(0).to(self.device)
                                        t_gt = torch.from_numpy(gt_arr).float().clone() / 127.5 - 1.0
                                        t_gt = t_gt.permute(2, 0, 1).unsqueeze(0).to(self.device)
                                        with torch.no_grad():
                                            dist = self.loss_fn(t_gen, t_gt)
                                            lpips_val = dist.item()
                                        lpips_score = max(0.0, 1.0 - lpips_val)
                                    except Exception as e:
                                        print(f"[RW Plugin] LPIPS Error: {e}")
                                        lpips_score = ssim_score
                                else:
                                    lpips_score = ssim_score

                                # 5. Final Weighted Score
                                # Use exponential mapping for MSE to provide better gradients
                                # exp(-10 * MSE) gives ~0.36 at MSE=0.1, ~0.9 at MSE=0.01
                                pixel_score_mapped = np.exp(-10.0 * pixel_mse)
                                edge_score = np.exp(-10.0 * edge_mse)

                                # Adjusted weights: Increased edge score importance for charts
                                raw_score = 0.4 * pixel_score_mapped + 0.15 * edge_score + 0.3 * ssim_score + 0.15 * lpips_score
                                
                                # 6. Final Score
                                visual_score = raw_score
                            
                            # print(f"[SVG Debug] Pixel: {pixel_score:.4f}, Grad: {grad_score:.4f}, Final: {visual_score:.4f}")
                        else:
                            # Avoid printing raw bytes or large dicts
                            debug_info = str(type(gt_path))
                            if isinstance(gt_path, str):
                                debug_info = gt_path
                            print(f"[SVG Reward Warning] GT image not found or invalid: {debug_info}")
                    else:
                        print(f"[SVG Reward Warning] No GT path found for idx {idx}, images len {len(images) if images else 0}")

                except Exception as e:
                    import traceback
                    print(f"[SVG Reward Visual Error] {str(e)}")
                    traceback.print_exc()
                    visual_score = 0.0

            # ---------- 3. Final Reward ----------
            reward = (
                self.semantic_weight * semantic_score +
                self.visual_weight * visual_score
            )
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            # Save debug image
            if gen_img is not None:
                try:
                    debug_dir = os.path.join(os.getcwd(), "grpo_generations_0203")
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    pid = os.getpid()
                    
                    # 1. Save Processed (Cropped) Prediction
                    if 'gen_img_processed' in locals() and gen_img_processed is not None:
                        processed_filename = f"{timestamp}_idx{idx}_pid{pid}_score_{reward:.2f}_vis_{visual_score:.2f}_processed.png"
                        gen_img_processed.save(os.path.join(debug_dir, processed_filename))
                        
                    # 2. Save Processed (Cropped) GT
                    should_save_gt = False
                    if 'gt_img' in locals() and gt_img is not None:
                        if idx == 0:
                            should_save_gt = True
                        else:
                            # Check if current GT path is different from previous
                            prev_gt_path = images[idx-1] if idx > 0 and idx < len(images) else None
                            curr_gt_path = images[idx] if idx < len(images) else None
                            if prev_gt_path != curr_gt_path:
                                should_save_gt = True
                        
                        if 'gt_img_processed' in locals() and gt_img_processed is not None and should_save_gt:
                             gt_processed_filename = f"{timestamp}_idx{idx}_pid{pid}_GT_processed.png"
                             gt_img_processed.save(os.path.join(debug_dir, gt_processed_filename))
                        
                except Exception as e:
                    print(f"[SVG Save Debug Error] {str(e)}")
                    pass

            print(
                f"[SVG Reward Debug] "
                f"Idx: {idx} | "
                f"Semantic: {semantic_score:.2f} | "
                f"Visual: {visual_score:.2f} | "
                f"Pixel: {pixel_score_mapped:.2f}, Edge: {edge_score:.2f}, SSIM: {ssim_score:.2f}, LPIPS: {lpips_score:.2f} | "
                f"Total: {reward:.4f}"
            )
            rewards.append(reward)

        return rewards

orms['svg_pipeline'] = SVGSemanticVisualReward


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# class AsyncGenRMReward(AsyncORM):
#     """
#     An async reward function example that calls a generative reward model
#     deployed via `swift deploy`.

#     This demonstrates how to use AsyncORM with aiohttp to make parallel API calls
#     to an LLM-based reward model for scoring completions.

#     The reward model is prompted to evaluate each completion and output a score
#     in a specific format (e.g., [[score]]).

#     Usage:
#         1. Deploy a reward model using swift deploy:
#            ```bash
#            swift deploy --model Qwen/Qwen2.5-7B-Instruct --port 8000 --infer_backend vllm
#            ```

#         2. Set environment variable:
#            ```bash
#            export GENRM_API_BASE=http://localhost:8000/v1
#            ```

#         3. Use in training:
#            ```bash
#            swift rlhf \
#                --rlhf_type grpo \
#                --external_plugins plugin.py \
#                --reward_funcs async_genrm ...
#            ```
#     """

#     def __init__(self):
#         from openai import OpenAI
#         self.api_base = os.getenv('GENRM_API_BASE', 'http://localhost:8000/v1')
#         self.temperature = float(os.getenv('GENRM_TEMPERATURE', '0.3'))

#         # Initialize OpenAI client to get the model name (following deepeyes_plugin pattern)
#         try:
#             self.client = OpenAI(
#                 api_key='EMPTY',
#                 base_url=self.api_base,
#             )
#             self.model_name = self.client.models.list().data[0].id
#             logger.info(f'AsyncGenRMReward initialized with model: {self.model_name}')
#         except Exception as e:
#             raise RuntimeError('Failed to connect to the model service. Please deploy the model '
#                                "using 'swift deploy --model <model_name> --port 8000 --infer_backend vllm'.") from e

#         # System prompt for the generative reward model
#         self.system_prompt = textwrap.dedent("""
#             You are an expert evaluator. Your task is to evaluate the quality of an AI assistant's response.

#             Please evaluate the response based on the following criteria:
#             1. Correctness: Is the answer factually correct and logically sound?
#             2. Helpfulness: Does the response address the user's question effectively?
#             3. Clarity: Is the response well-organized and easy to understand?

#             After your evaluation, provide a score from 0 to 10, where:
#             - 0-3: Poor quality (incorrect, unhelpful, or confusing)
#             - 4-6: Acceptable quality (partially correct or helpful)
#             - 7-9: Good quality (correct, helpful, and clear)
#             - 10: Excellent quality (perfect response)

#             You MUST end your response with the score in this exact format: [[score]]
#             For example: [[7]] or [[10]]
#         """).strip()

#     def _build_eval_prompt(self, question: str, completion: str) -> str:
#         """Build the evaluation prompt for the reward model."""
#         return textwrap.dedent(f"""
#             ## User Question
#             {question}

#             ## AI Assistant's Response
#             {completion}

#             ## Your Evaluation
#             Please evaluate the above response and provide your score.
#         """).strip()

#     def _extract_score(self, response: str) -> float:
#         """Extract the score from the reward model's response."""
#         # Look for [[score]] pattern
#         match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
#         if match:
#             score = float(match.group(1))
#             # Normalize to [0, 1] range
#             return min(max(score / 10.0, 0.0), 1.0)

#         # Fallback: try to find any number at the end
#         match = re.search(r'(\d+(?:\.\d+)?)\s*$', response.strip())
#         if match:
#             score = float(match.group(1))
#             return min(max(score / 10.0, 0.0), 1.0)

#         logger.warning(f'Could not extract score from response: {response[:100]}...')
#         return 0.0

#     async def _score_single(self, session, question: str, completion: str) -> float:
#         """Score a single completion using the generative reward model."""
#         import aiohttp

#         eval_prompt = self._build_eval_prompt(question, completion)

#         payload = {
#             'model': self.model_name,
#             'messages': [{
#                 'role': 'system',
#                 'content': self.system_prompt
#             }, {
#                 'role': 'user',
#                 'content': eval_prompt
#             }],
#             'temperature': self.temperature,
#             'max_tokens': 2048,
#             'seed': random.randint(0, 1000000),
#         }

#         try:
#             async with session.post(
#                     f'{self.api_base}/chat/completions', json=payload,
#                     timeout=aiohttp.ClientTimeout(total=120)) as resp:
#                 if resp.status != 200:
#                     error_text = await resp.text()
#                     logger.warning(f'API error {resp.status}: {error_text[:200]}')
#                     return 0.0

#                 result = await resp.json()
#                 response_content = result['choices'][0]['message']['content']
#                 return self._extract_score(response_content)

#         except asyncio.TimeoutError:
#             logger.warning('API request timed out')
#             return 0.0
#         except Exception as e:
#             logger.warning(f'Error calling reward model API: {e}')
#             return 0.0

#     async def __call__(self, completions, messages, **kwargs) -> List[float]:
#         """
#         Score completions using a generative reward model via async API calls.

#         Args:
#             completions: List of model-generated responses
#             messages: List of conversation messages (used to extract the question)
#             **kwargs: Additional arguments (unused)

#         Returns:
#             List of reward scores in [0, 1] range
#         """
#         import aiohttp

#         # Extract questions from messages (assuming the last user message is the question)
#         questions = []
#         for msg_list in messages:
#             question = ''
#             for msg in reversed(msg_list):
#                 if msg.get('role') == 'user':
#                     question = msg.get('content', '')
#                     break
#             questions.append(question)

#         # Make parallel API calls using asyncio.gather
#         async with aiohttp.ClientSession() as session:
#             tasks = [self._score_single(session, q, c) for q, c in zip(questions, completions)]
#             rewards = await asyncio.gather(*tasks)
#             return list(rewards)


# orms['async_genrm'] = AsyncGenRMReward


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
