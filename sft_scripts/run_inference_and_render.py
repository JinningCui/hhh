import sys
import os
import json
import re
import torch
from datetime import datetime
from lxml import etree
from pathlib import Path
from tqdm import tqdm
import cairosvg
from PIL import Image
import argparse

# Add project root to sys.path to allow importing Chart2SVG modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from swift.llm import infer_main, InferArguments
from swift.llm.infer import SwiftInfer
from Chart2SVG.data.semantic_tokens import syntactic2svg

TARGET_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and render SVG")
    parser.add_argument('--checkpoint_path', type=str, default='/home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save rendered SVGs/PNGs')
    parser.add_argument('--resized_dir', type=str, required=True, help='Directory to save resized images')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process (for debugging)')
    parser.add_argument('--model_type', type=str, default='qwen3_vl', help='Model type (e.g. qwen3_vl, qwen2-vl). Default is qwen3_vl.')
    return parser.parse_args()

def normalize_image(image_path, target_size=TARGET_SIZE):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        scale = target_size / max(w, h)
        w_new = int(w * scale)
        h_new = int(h * scale)
        img_resized = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
        pad_x = (target_size - w_new) / 2
        pad_y = (target_size - h_new) / 2
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        new_img.paste(img_resized, (int(pad_x), int(pad_y)))
        return new_img, scale, pad_x, pad_y, w, h

_PATH_TOKEN_RE = re.compile(r'([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def _expand_arc_flags(tokens):
    new_tokens = []
    i = 0
    n = len(tokens)
    current_cmd = None
    arg_idx = 0 
    
    while i < n:
        t = tokens[i]
        if t[0].isalpha() and t.upper() not in ['E', 'e']:
            current_cmd = t.upper()
            new_tokens.append(t)
            arg_idx = 0
            i += 1
            continue
            
        if current_cmd == 'A':
            cycle = arg_idx % 7
            if cycle in (3, 4):
                if t == "0" or t == "1":
                    new_tokens.append(t)
                    arg_idx += 1
                    i += 1
                else:
                    first = t[0]
                    remainder = t[1:]
                    if first in ('0', '1'):
                        new_tokens.append(first)
                        arg_idx += 1
                        if remainder:
                            tokens[i] = remainder
                        else:
                            i += 1
                    else:
                        new_tokens.append(t)
                        arg_idx += 1
                        i += 1
            else:
                new_tokens.append(t)
                arg_idx += 1
                i += 1
        else:
            new_tokens.append(t)
            i += 1
            
    return new_tokens

def untransform_path_d(d, scale):
    if not d or len(d) < 2:
        return d

    # Pre-process
    d = d.replace('NaN', '0').replace('nan', '0').replace('NAN', '0').replace('null', '0')

    tokens = _PATH_TOKEN_RE.findall(d)
    
    raw_tokens = []
    for t in tokens:
        if t[0]: raw_tokens.append(t[0])
        else: raw_tokens.append(t[1])
        
    flat_tokens_str = _expand_arc_flags(raw_tokens)
    
    flat_tokens = []
    for t in flat_tokens_str:
        if t[0].isalpha() and t.upper() not in ['E', 'e']:
             flat_tokens.append(t)
        else:
             try:
                 flat_tokens.append(float(t))
             except ValueError:
                 flat_tokens.append(0.0)
            
    res = []
    n_tokens = len(flat_tokens)
    i = 0
    current_cmd = None
    
    args_count = {
        'M': 2, 'm': 2, 'L': 2, 'l': 2, 'H': 1, 'h': 1, 'V': 1, 'v': 1,
        'C': 6, 'c': 6, 'S': 4, 's': 4, 'Q': 4, 'q': 4, 'T': 2, 't': 2,
        'A': 7, 'a': 7, 'Z': 0, 'z': 0
    }
    
    while i < n_tokens:
        token = flat_tokens[i]
        if isinstance(token, str):
            current_cmd = token
            res.append(current_cmd)
            i += 1
        
        if current_cmd is None:
            i += 1
            continue
            
        cmd_upper = current_cmd.upper()
        n_args = args_count.get(cmd_upper, 0)
        
        if n_args == 0:
            i += 1
            continue
            
        start_arg_idx = i
        end_arg_idx = i
        while end_arg_idx < n_tokens and not isinstance(flat_tokens[end_arg_idx], str):
            end_arg_idx += 1
            
        available_args = end_arg_idx - start_arg_idx
        
        # Process arguments in chunks of n_args
        for chunk_start in range(0, available_args, n_args):
            if chunk_start + n_args > available_args:
                break
                
            args = flat_tokens[start_arg_idx + chunk_start : start_arg_idx + chunk_start + n_args]
            new_args = []
            
            # Helper to unscale
            def unscale(val):
                return round(val / scale, 2)
            
            if cmd_upper == 'A':
                # rx ry x-axis-rotation large-arc-flag sweep-flag x y
                new_args.append(unscale(args[0])) # rx
                new_args.append(unscale(args[1])) # ry
                new_args.append(args[2])          # rot
                new_args.append(args[3])          # large
                new_args.append(args[4])          # sweep
                new_args.append(unscale(args[5])) # x
                new_args.append(unscale(args[6])) # y
            elif cmd_upper in ['H', 'h', 'V', 'v']:
                new_args.append(unscale(args[0]))
            else:
                new_args = [unscale(x) for x in args]
                
            res.extend(map(str, new_args))
            
        i = end_arg_idx

    return " ".join(res)

def _unscale_transform_str(transform_str, scale):
    if not transform_str:
        return transform_str
    
    pattern = re.compile(r'([a-zA-Z]+)\s*\(([^)]*)\)')
    
    def repl(match):
        cmd = match.group(1).lower()
        args_str = match.group(2)
        raw_args = re.split(r'[,\s]+', args_str.strip())
        args = []
        for x in raw_args:
            if not x: continue
            clean_x = x.lower().replace('deg', '').replace('px', '').replace('null', '0')
            try:
                args.append(float(clean_x))
            except ValueError:
                args.append(0.0)
        
        if not args:
            return match.group(0)

        new_args = []
        
        if cmd == 'translate':
            new_args.append(round(args[0] / scale, 2))
            if len(args) > 1:
                new_args.append(round(args[1] / scale, 2))
                
        elif cmd == 'matrix':
            if len(args) == 6:
                new_args = [
                    args[0], args[1], args[2], args[3],
                    round(args[4] / scale, 2), # e (tx)
                    round(args[5] / scale, 2)  # f (ty)
                ]
            else:
                return match.group(0)
                
        elif cmd == 'rotate':
            new_args.append(args[0])
            if len(args) > 1:
                new_args.append(round(args[1] / scale, 2))
            if len(args) > 2:
                new_args.append(round(args[2] / scale, 2))
                
        else:
            new_args = [round(x, 2) for x in args] # keep others? or unscale? 
            # Scale usually means scaling the object, if we are restoring coordinate system,
            # we don't change object local scale.
            
        args_joined = ", ".join(map(str, new_args))
        return f"{match.group(1)}({args_joined})"

    return pattern.sub(repl, transform_str)

def _unscale_style_str(style_str, scale):
    if not style_str:
        return style_str
    
    parts = [p.strip() for p in style_str.split(';') if p.strip()]
    new_parts = []
    
    for part in parts:
        if ':' not in part:
            new_parts.append(part)
            continue
            
        key, val = [x.strip() for x in part.split(':', 1)]
        key_lower = key.lower()
        
        if key_lower in ('font-size', 'stroke-width'):
            try:
                val_clean = val.lower().replace('px', '').strip()
                val_float = float(val_clean)
                new_val = round(val_float / scale, 2)
                new_parts.append(f"{key}: {new_val}px")
            except ValueError:
                new_parts.append(part)
        else:
            new_parts.append(part)
            
    return "; ".join(new_parts)

def extract_svg_code(text):
    # Try to find SVG block
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Strip markdown code blocks if no clear svg tag found but markdown is present
    text = re.sub(r'```(?:xml|svg)?\n', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text.strip()

def denormalize_svg(svg_code, meta):
    if not svg_code or not meta:
        return svg_code
        
    orig_w = int(meta['orig_w'])
    orig_h = int(meta['orig_h'])
    scale = float(meta['scale'])
    pad_x = float(meta['pad_x'])
    pad_y = float(meta['pad_y'])
    
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    try:
        root = etree.fromstring(svg_code.encode('utf-8'), parser)
    except Exception:
        return svg_code
        
    if root is None:
        return svg_code
        
    # Attempt to find the actual content offset from the SVG
    # The model often generates a group with translate(x, y) that represents the padding
    actual_tx = pad_x
    actual_ty = pad_y
    
    found_translate = False
    # Heuristic: Scan for a group with translate that is close to the expected padding
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem.tag.endswith('g') and 'transform' in elem.attrib:
            t = elem.attrib['transform']
            if 'translate' in t:
                nums = re.findall(r'translate\s*\(\s*([-+]?\d*\.?\d+)\s*(?:[,\s]\s*([-+]?\d*\.?\d+))?\s*\)', t)
                if nums:
                    tx = float(nums[0][0])
                    ty = float(nums[0][1]) if nums[0][1] else 0.0
                    
                    # Check closeness (e.g. within 50px) to expected padding
                    # This helps distinguish the main wrapper from other translations
                    if abs(tx - pad_x) < 50 and abs(ty - pad_y) < 50:
                        actual_tx = tx
                        actual_ty = ty
                        found_translate = True
                        break # Assume the first matching group is the main wrapper

    # Use the detected translation as the start of the content
    # This ensures we don't cut off content if the model generated slightly different padding
    start_x = actual_tx
    start_y = actual_ty
    
    # Calculate viewBox dimensions assuming symmetry (centered content)
    # This mirrors how padding was added: pad = (TARGET_SIZE - new_dim) / 2
    # So new_dim = TARGET_SIZE - 2 * pad
    w_view = TARGET_SIZE - start_x * 2
    h_view = TARGET_SIZE - start_y * 2
    
    # Fallback/Safety checks
    if w_view <= 0: w_view = orig_w * scale
    if h_view <= 0: h_view = orig_h * scale
    
    # Set viewBox to the detected content region
    root.attrib['viewBox'] = f"{start_x} {start_y} {w_view} {h_view}"
    
    # Set the display dimensions to the original image size
    root.attrib['width'] = str(orig_w)
    root.attrib['height'] = str(orig_h)
    
    return etree.tostring(root, encoding='unicode', pretty_print=False)


def run_inference(args):
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    output_dir = Path(args.output_dir)
    resized_dir = Path(args.resized_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
        
    print(f"Found {len(data)} samples.")
    
    if args.limit:
        print(f"Limiting to first {args.limit} samples for debugging.")
        data = data[:args.limit]

    # 1.1 Normalize Images and record meta
    print("Normalizing images to 512x512...")
    meta_map = {}
    
    for item in tqdm(data, desc="Resizing"):
        images = item.get('images', [])
        new_images = []
        for img_path in images:
            src = Path(img_path)
            if not src.exists():
                new_images.append(img_path)
                continue
            new_filename = f"{src.stem}_normalized{src.suffix}"
            dst = resized_dir / new_filename
            if not dst.exists():
                try:
                    norm_img, scale, pad_x, pad_y, w0, h0 = normalize_image(src)
                    norm_img.save(dst)
                    meta_map[str(dst)] = {'orig_w': w0, 'orig_h': h0, 'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y}
                except Exception:
                    new_images.append(img_path)
                    continue
            else:
                if str(dst) not in meta_map:
                    try:
                        with Image.open(src) as im:
                            im = im.convert('RGB')
                            w0, h0 = im.size
                        s = TARGET_SIZE / max(w0, h0)
                        w_new = int(w0 * s)
                        h_new = int(h0 * s)
                        meta_map[str(dst)] = {
                            'orig_w': w0,
                            'orig_h': h0,
                            'scale': s,
                            'pad_x': (TARGET_SIZE - w_new) / 2,
                            'pad_y': (TARGET_SIZE - h_new) / 2
                        }
                    except Exception:
                        pass
            new_images.append(str(dst))
        item['images'] = new_images
        
    # Update dataset json
    updated_dataset_path = Path(dataset_path).with_name(f"{Path(dataset_path).stem}_processed.json")
    with open(updated_dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved resized dataset to {updated_dataset_path}")

    # 2. Configure SwiftInfer
    ckpt_path_obj = Path(checkpoint_path)
    if ckpt_path_obj.is_dir() and not (ckpt_path_obj / 'adapter_config.json').exists():
        candidate_ckpts = [d for d in ckpt_path_obj.iterdir() if d.is_dir() and d.name.startswith('checkpoint')]
        if candidate_ckpts:
            def _step(x):
                try:
                    return int(x.name.split('-')[-1])
                except Exception:
                    return 0
            latest = sorted(candidate_ckpts, key=_step)[-1]
            checkpoint_path = str(latest)
            print(f"Detected run directory, using latest checkpoint: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    is_lora = (Path(checkpoint_path) / 'adapter_config.json').exists()
    
    adapters = []
    base_model = checkpoint_path
    
    if is_lora:
        with open(Path(checkpoint_path) / 'adapter_config.json', 'r') as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get('base_model_name_or_path')
        adapters = [checkpoint_path]
        print(f"Detected LoRA checkpoint. Base model: {base_model}")
    
    infer_args = InferArguments(
        model=base_model,
        adapters=adapters,
        val_dataset=[str(updated_dataset_path)],
        max_new_tokens=8192,
        temperature=0.5,
        repetition_penalty=1.1,
        stream=True,
        max_batch_size=4,
        model_type=args.model_type
    )
    
    # 3. Run Inference
    print("Running inference with SwiftInfer...")
    results = SwiftInfer(infer_args).run()

    # 4. Process Results
    print("Rendering SVGs...")
    failed_samples = []
    
    for i, item in enumerate(tqdm(results, desc="Rendering")):
        response = item['response']
        images = item.get('images', [])
        base_name = f"sample_{i}" # Default fallback
        
        try:
            # response is semantic svg tokens
            semantic_svg = response
            try:
                # Convert semantic tokens to standard SVG XML
                standard_svg = syntactic2svg(semantic_svg)
            except Exception as e:
                print(f"Error in syntactic2svg for {base_name}: {e}")
                standard_svg = "" # Or handle gracefully
            
            svg_code = standard_svg
            
            # Define output paths
            img_key = None
            if images:
                img_info = images[0]
                if isinstance(img_info, str):
                    img_key = img_info
                elif isinstance(img_info, dict) and 'path' in img_info:
                    img_key = img_info['path']

            if img_key:
                base_name = Path(img_key).stem
            
            svg_path = output_dir / f"{base_name}.svg"
            png_path = output_dir / f"{base_name}.png"
            
            meta = None
            if img_key and img_key in meta_map:
                meta = meta_map[img_key]
            
            final_svg = svg_code
            if meta:
                # Use denormalize_svg to restore original dimensions and coordinates
                final_svg = denormalize_svg(svg_code, meta)

            # Save JSON with history
            json_path = output_dir / f"{base_name}.json"
            current_entry = {
                "timestamp": datetime.now().isoformat(),
                "semantic_svg": response,
                "standard_svg": svg_code,
                "converted_svg": final_svg
            }
            
            existing_data = []
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data] # Handle legacy non-list format if any
                except Exception as e:
                    print(f"Warning: Could not read existing JSON {json_path}: {e}")
            
            existing_data.append(current_entry)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

            with open(svg_path, 'w') as f:
                f.write(final_svg)
            cairosvg.svg2png(bytestring=final_svg.encode('utf-8'), write_to=str(png_path))
            
        except Exception as e:
            print(f"Error processing sample {i} ({base_name}): {e}")
            failed_samples.append({'index': i, 'name': base_name, 'error': str(e)})

    print(f"\nProcessing completed. Total: {len(results)}, Failed: {len(failed_samples)}")
    if failed_samples:
        print("Failures:")
        for fail in failed_samples:
            print(f"  - Sample {fail['index']} ({fail['name']}): {fail['error']}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
    # response = ""
    # # try:
    #     svg_code = syntactic2svg(response)
            
    #     # Define output paths
    #     base_name = f"sample"
    #     svg_path = output_dir / f"{base_name}.svg"
    #     png_path = output_dir / f"{base_name}.png"
    #     meta = None
    #     with open(svg_path, 'w') as f:
    #         f.write(svg_code)
    #     cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), write_to=str(png_path))
            
    # except Exception as e:
    #     print(f"Error processing sample: {e}")
