import sys
import os
import json
import re
import torch
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

# from swift.llm import infer_main, InferArguments
# from swift.llm.infer import SwiftInfer
from Chart2SVG.data.semantic_tokens import syntactic2svg

TARGET_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and render SVG")
    parser.add_argument('--checkpoint_path', type=str, default='/home/u20249114/ms-swift/svg_output/v12-20260102-163107/checkpoint-3189')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save rendered SVGs/PNGs')
    parser.add_argument('--resized_dir', type=str, required=True, help='Directory to save resized images')
    parser.add_argument('--limit', type=int, default=2, help='Limit the number of samples to process (for debugging)')
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
        
    # Attempt to find the actual content offset from the SVG
    # The model often generates a group with translate(x, y) that represents the padding
    actual_tx = pad_x
    actual_ty = pad_y
    
    found_translate = False
    # Heuristic: Scan for a group with translate that is close to the expected padding
    for elem in root.iter():
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
        max_new_tokens=16384,
        temperature=0,
        stream=True,
        max_batch_size=1,
        model_type='qwen3_vl' # Hint for Swift if needed
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
            svg_code = syntactic2svg(response)
            
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
            
            if meta:
                # Use denormalize_svg to restore original dimensions and coordinates
                svg_code = denormalize_svg(svg_code, meta)

            with open(svg_path, 'w') as f:
                f.write(svg_code)
            cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), write_to=str(png_path))
            
        except Exception as e:
            print(f"Error processing sample {i} ({base_name}): {e}")
            failed_samples.append({'index': i, 'name': base_name, 'error': str(e)})

    print(f"\nProcessing completed. Total: {len(results)}, Failed: {len(failed_samples)}")
    if failed_samples:
        print("Failures:")
        for fail in failed_samples:
            print(f"  - Sample {fail['index']} ({fail['name']}): {fail['error']}")

if __name__ == "__main__":
    # args = parse_args()
    # run_inference(args)
    response = "[<|START_OF_SVG|>][<|viewBox=|>]0 0 512 512[<|width=|>]512[<|height=|>]512[<|rect|>][<|width=|>]512[<|height=|>]512[<|x=|>]0[<|y=|>]0[<|fill=|>]white[<|/rect|>][<|START_OF_GROUP|>][<|fill=|>]none[<|stroke-width=|>]0.73px[<|transform=|>]translate(0.0, 73.5)[<|font-size=|>]5.85px[<|rect|>][<|fill=|>]rgb(255, 255, 255)[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|x=|>]0[<|y=|>]0[<|width=|>]512.0[<|height=|>]365.71[<|fill-opacity=|>]1[<|/rect|>][<|defs|>][<|START_OF_GROUP|>][<|class=|>]clips[<|clipPath|>][<|rect|>][<|fill=|>]none[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|x=|>]58.51[<|y=|>]0[<|width=|>]446.17[<|height=|>]365.71[<|/rect|>][<|/clipPath|>][<|clipPath|>][<|rect|>][<|fill=|>]none[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|x=|>]0[<|y=|>]58.51[<|width=|>]512.0[<|height=|>]248.69[<|/rect|>][<|/clipPath|>][<|clipPath|>][<|rect|>][<|fill=|>]none[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|x=|>]58.51[<|y=|>]58.51[<|width=|>]446.17[<|height=|>]248.69[<|/rect|>][<|/clipPath|>][<|clipPath|>][<|id=|>]clip01200cxyplot[<|rect|>][<|fill=|>]none[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|width=|>]446.17[<|height=|>]248.69[<|/rect|>][<|/clipPath|>][<|END_OF_GROUP|>][<|/defs|>][<|START_OF_GROUP|>][<|class=|>]layer-below[<|START_OF_GROUP|>][<|class=|>]imagelayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]shapelayer[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]subplot xy[<|rect|>][<|fill=|>]rgb(255, 255, 255)[<|stroke-width=|>]0[<|font-size=|>]5.85[<|x=|>]58.51[<|y=|>]58.51[<|width=|>]446.17[<|height=|>]248.69[<|fill-opacity=|>]1[<|/rect|>][<|START_OF_GROUP|>][<|class=|>]layer-subplot[<|START_OF_GROUP|>][<|class=|>]shapelayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]imagelayer[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(58.51, 58.51)[<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|vertical_lineto_rel|>]248.69[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(22.49, 0)[<|class=|>]xgrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|vertical_lineto_rel|>]248.69[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(122.69, 0)[<|class=|>]xgrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|vertical_lineto_rel|>]248.69[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(222.88, 0)[<|class=|>]xgrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|vertical_lineto_rel|>]248.69[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(323.08, 0)[<|class=|>]xgrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|vertical_lineto_rel|>]248.69[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(423.27, 0)[<|class=|>]xgrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 223.28)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 186.45)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 149.62)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 112.79)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 75.96)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 39.13)[<|class=|>]ygrid crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]0 0[<|horizontal_lineto_rel|>]446.17[<|fill=|>]none[<|stroke=|>]rgb(221, 221, 221)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]1[<|transform=|>]translate(0, 2.3)[<|class=|>]ygrid crisp[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(58.51, 58.51)[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(58.51, 58.51)[<|clip-path=|>]url(#clip01200cxyplot)[<|START_OF_GROUP|>][<|class=|>]imagelayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]maplayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]barlayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]boxlayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]scatterlayer[<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]trace scatter[<|path|>][<|d=|>][<|moveto_abs|>]22.49 223.28[<|lineto_abs|>]122.69 186.45[<|lineto_abs|>]222.88 168.01[<|lineto_abs|>]323.08 153.91[<|lineto_abs|>]423.27 88.29[<|fill=|>]none[<|stroke=|>]rgb(31, 119, 180)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]trace scatter[<|path|>][<|d=|>][<|moveto_abs|>]22.49 56.39[<|lineto_abs|>]122.69 122.02[<|lineto_abs|>]222.88 41.47[<|lineto_abs|>]323.08 29.93[<|lineto_abs|>]423.27 7.58[<|fill=|>]none[<|stroke=|>]rgb(255, 127, 14)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]trace scatter[<|path|>][<|d=|>][<|moveto_abs|>]22.49 137.11[<|lineto_abs|>]122.69 107.61[<|lineto_abs|>]222.88 137.11[<|lineto_abs|>]323.08 71.49[<|lineto_abs|>]423.27 39.13[<|fill=|>]none[<|stroke=|>]rgb(44, 160, 44)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|END_OF_GROUP|>][<|path|>][<|d=|>][<|moveto_abs|>]-0.73 249.06[<|horizontal_lineto_rel|>]447.63[<|fill=|>]none[<|stroke=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]0[<|transform=|>]translate(58.51, 58.51)[<|class=|>]crisp[<|/path|>][<|path|>][<|d=|>][<|moveto_abs|>]-0.37 -0.73[<|vertical_lineto_rel|>]249.42[<|fill=|>]none[<|stroke=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|stroke-opacity=|>]0[<|transform=|>]translate(58.51, 58.51)[<|class=|>]crisp[<|/path|>][<|START_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(58.51, 58.51)[<|START_OF_GROUP|>][<|class=|>]xtick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]start[<|x=|>]0[<|y=|>]264.07[<|transform=|>]translate(22.49, 0) rotate(30, 0, 256.77)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]2011[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]xtick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]start[<|x=|>]0[<|y=|>]264.07[<|transform=|>]translate(122.69, 0) rotate(30, 0, 256.77)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]2012[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]xtick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]start[<|x=|>]0[<|y=|>]264.07[<|transform=|>]translate(222.88, 0) rotate(30, 0, 256.77)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]2013[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]xtick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]start[<|x=|>]0[<|y=|>]264.07[<|transform=|>]translate(323.08, 0) rotate(30, 0, 256.77)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]2014[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]xtick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]start[<|x=|>]0[<|y=|>]264.07[<|transform=|>]translate(423.27, 0) rotate(30, 0, 256.77)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]2015[<|/text|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(58.51, 58.51)[<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 223.28)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]100[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 186.45)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]120[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 149.62)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]140[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 112.79)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]160[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 75.96)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]180[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 39.13)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]200[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ytick[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(68, 68, 68); fill-opacity: 1; visibility: visible; white-space: pre[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|text-anchor=|>]end[<|x=|>]-1.46[<|y=|>]6.58[<|transform=|>]translate(0, 2.3)[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-content=|>]220[<|/text|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]ternarylayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]layer-above[<|START_OF_GROUP|>][<|class=|>]imagelayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]shapelayer[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]pielayer[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]glimages[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]geoimages[<|END_OF_GROUP|>][<|text|>][<|style=|>]font-family: 'Open Sans', Arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); pointer-events: all[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|class=|>]js-plot-link-container[<|y=|>]359.13[<|text-anchor=|>]end[<|x=|>]506.88[<|font-family=|>]'Open Sans', Arial, sans-serif[<|tspan|>][<|style=|>]font-family: 'Open Sans', Arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); pointer-events: all[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|text-anchor=|>]end[<|font-family=|>]'Open Sans', Arial, sans-serif[<|class=|>]js-link-to-tool[<|/tspan|>][<|tspan|>][<|style=|>]font-family: 'Open Sans', Arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); pointer-events: all[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|text-anchor=|>]end[<|font-family=|>]'Open Sans', Arial, sans-serif[<|class=|>]js-link-spacer[<|/tspan|>][<|tspan|>][<|style=|>]font-family: 'Open Sans', Arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); pointer-events: all[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|text-anchor=|>]end[<|font-family=|>]'Open Sans', Arial, sans-serif[<|class=|>]js-sourcelinks[<|/tspan|>][<|/text|>][<|defs|>][<|clipPath|>][<|id=|>]legend01200c[<|rect|>][<|fill=|>]none[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|width=|>]191.19[<|height=|>]62.9[<|x=|>]0[<|y=|>]0[<|/rect|>][<|/clipPath|>][<|/defs|>][<|START_OF_GROUP|>][<|class=|>]infolayer[<|START_OF_GROUP|>][<|class=|>]g-gtitle[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 14.63px; fill: rgb(0, 0, 0); opacity: 1; font-weight: normal; white-space: pre; visibility: visible[<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]14.63[<|class=|>]gtitle[<|x=|>]256[<|y=|>]36.57[<|text-anchor=|>]middle[<|opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|font-weight=|>]normal[<|text-content=|>]Production in manufacturing industry from 2011 to 2015[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(0.0, -1.46)[<|class=|>]g-xtitle[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(0, 0, 0); opacity: 1; font-weight: normal; white-space: pre; visibility: visible[<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|class=|>]xtitle[<|x=|>]281.6[<|y=|>]340.85[<|text-anchor=|>]middle[<|opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|font-weight=|>]normal[<|text-content=|>]Year[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]g-ytitle[<|text|>][<|style=|>]font-family: 'Open sans', verdana, arial, sans-serif; font-size: 13.17px; fill: rgb(0, 0, 0); opacity: 1; font-weight: normal; white-space: pre; visibility: visible[<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]13.17[<|class=|>]ytitle[<|transform=|>]rotate(-90, 33.65, 190.17) translate(0, 0)[<|x=|>]33.65[<|y=|>]190.17[<|text-anchor=|>]middle[<|opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|font-weight=|>]normal[<|text-content=|>]Units (million)[<|/text|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|transform=|>]translate(65.01, 65.01)[<|class=|>]legend[<|rect|>][<|fill=|>]rgb(255, 255, 255)[<|stroke-width=|>]0[<|font-size=|>]5.85[<|class=|>]bg[<|width=|>]191.19[<|height=|>]62.9[<|x=|>]0[<|y=|>]0[<|stroke=|>]rgb(0, 0, 0)[<|fill-opacity=|>]1[<|stroke-opacity=|>]0[<|/rect|>][<|START_OF_GROUP|>][<|transform=|>]translate(0.0, 0.0)[<|class=|>]scrollbox[<|clip-path=|>]url(#legend01200c)[<|START_OF_GROUP|>][<|class=|>]groups[<|START_OF_GROUP|>][<|opacity=|>]1[<|transform=|>]translate(0.0, 13.31)[<|class=|>]traces[<|START_OF_GROUP|>][<|class=|>]legendfill[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]legendlines[<|path|>][<|d=|>][<|moveto_abs|>]3.66 0[<|horizontal_lineto_rel|>]21.94[<|fill=|>]none[<|stroke=|>]rgb(31, 119, 180)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]legendsymbols[<|START_OF_GROUP|>][<|class=|>]legendpoints[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|text|>][<|style=|>]text-anchor: start; font-family: 'Open sans', verdana, arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre; visibility: visible[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|opacity=|>]1[<|class=|>]legendtext user-select-none[<|x=|>]29.26[<|y=|>]3.99[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-anchor=|>]start[<|text-content=|>]Production A (million units)[<|/text|>][<|rect|>][<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|opacity=|>]1[<|class=|>]legendtoggle[<|width=|>]244.57[<|x=|>]0[<|y=|>]-7.75[<|height=|>]15.5[<|fill-opacity=|>]0[<|/rect|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|transform=|>]translate(0.0, 31.44)[<|class=|>]traces[<|START_OF_GROUP|>][<|class=|>]legendfill[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]legendlines[<|path|>][<|d=|>][<|moveto_abs|>]3.66 0[<|horizontal_lineto_rel|>]21.94[<|fill=|>]none[<|stroke=|>]rgb(255, 127, 14)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]legendsymbols[<|START_OF_GROUP|>][<|class=|>]legendpoints[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|text|>][<|style=|>]text-anchor: start; font-family: 'Open sans', verdana, arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre; visibility: visible[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|opacity=|>]1[<|class=|>]legendtext user-select-none[<|x=|>]29.26[<|y=|>]3.99[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-anchor=|>]start[<|text-content=|>]Production B (million units)[<|/text|>][<|rect|>][<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|opacity=|>]1[<|class=|>]legendtoggle[<|width=|>]244.57[<|x=|>]0[<|y=|>]-7.75[<|height=|>]15.5[<|fill-opacity=|>]0[<|/rect|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|transform=|>]translate(0.0, 49.58)[<|class=|>]traces[<|START_OF_GROUP|>][<|class=|>]legendfill[<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|class=|>]legendlines[<|path|>][<|d=|>][<|moveto_abs|>]3.66 0[<|horizontal_lineto_rel|>]21.94[<|fill=|>]none[<|stroke=|>]rgb(44, 160, 44)[<|stroke-width=|>]1.46[<|opacity=|>]1[<|stroke-opacity=|>]1[<|class=|>]js-line[<|/path|>][<|END_OF_GROUP|>][<|START_OF_GROUP|>][<|opacity=|>]1[<|class=|>]legendsymbols[<|START_OF_GROUP|>][<|class=|>]legendpoints[<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|text|>][<|style=|>]text-anchor: start; font-family: 'Open sans', verdana, arial, sans-serif; font-size: 8.78px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre; visibility: visible[<|fill=|>]rgb(68, 68, 68)[<|stroke-width=|>]0.73[<|font-size=|>]8.78[<|opacity=|>]1[<|class=|>]legendtext user-select-none[<|x=|>]29.26[<|y=|>]3.99[<|fill-opacity=|>]1[<|font-family=|>]'Open sans', verdana, arial, sans-serif[<|text-anchor=|>]start[<|text-content=|>]Production C (million units)[<|/text|>][<|rect|>][<|fill=|>]rgb(0, 0, 0)[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|opacity=|>]1[<|class=|>]legendtoggle[<|width=|>]244.57[<|x=|>]0[<|y=|>]-7.75[<|height=|>]15.5[<|fill-opacity=|>]0[<|/rect|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|rect|>][<|fill=|>]rgb(128, 139, 164)[<|stroke-width=|>]0.73[<|font-size=|>]5.85[<|class=|>]scrollbar[<|rx=|>]14.63[<|ry=|>]1.46[<|width=|>]0.0[<|height=|>]0.0[<|fill-opacity=|>]1[<|/rect|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|END_OF_GROUP|>][<|END_OF_SVG|>]"
    output_dir = Path("/home/u20249114/ms-swift/inference_results")
    try:
        svg_code = syntactic2svg(response)
        print(svg_code)
        
        # Define output paths
        base_name = f"sample_line"
        svg_path = output_dir / f"{base_name}.svg"
        png_path = output_dir / f"{base_name}.png"
        meta = None
        with open(svg_path, 'w') as f:
            f.write(svg_code)
        cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), write_to=str(png_path))
            
    except Exception as e:
        print(f"Error processing sample: {e}")
