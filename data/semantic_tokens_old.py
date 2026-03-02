# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: SVG Semantic Tokens encoder and decoder
# Fixed version for Chart2SVG task

import re
from typing import Union, List, Tuple

from lxml import etree
from xml.sax.saxutils import escape

"""Define SVG Mappers and Identifiers"""

PathCMDMapper = {
    '[m]': '[<|moveto_rel|>]',
    '[M]': '[<|moveto_abs|>]',
    '[l]': '[<|lineto_rel|>]',
    '[L]': '[<|lineto_abs|>]',
    '[h]': '[<|horizontal_lineto_rel|>]',
    '[H]': '[<|horizontal_lineto_abs|>]',
    '[v]': '[<|vertical_lineto_rel|>]',
    '[V]': '[<|vertical_lineto_abs|>]',
    '[c]': '[<|curveto_rel|>]',
    '[C]': '[<|curveto_abs|>]',
    '[s]': '[<|smooth_curveto_rel|>]',
    '[S]': '[<|smooth_curveto_abs|>]',
    '[q]': '[<|quadratic_bezier_curve_rel|>]',
    '[Q]': '[<|quadratic_bezier_curve_abs|>]',
    '[t]': '[<|smooth_quadratic_bezier_curveto_rel|>]',
    '[T]': '[<|smooth_quadratic_bezier_curveto_abs|>]',
    '[a]': '[<|elliptical_arc_rel|>]',
    '[A]': '[<|elliptical_arc_abs|>]',
    '[z]': '[<|close_path|>]',
    '[Z]': '[<|close_path|>]',
}

PathCMDIdentifier = {
    'moveto_rel': 'm',
    'moveto_abs': 'M',
    'lineto_rel': 'l',
    'lineto_abs': 'L',
    'horizontal_lineto_rel': 'h',
    'horizontal_lineto_abs': 'H',
    'vertical_lineto_rel': 'v',
    'vertical_lineto_abs': 'V',
    'curveto_rel': 'c',
    'curveto_abs': 'C',
    'smooth_curveto_rel': 's',
    'smooth_curveto_abs': 'S',
    'quadratic_bezier_curve_rel': 'q',
    'quadratic_bezier_curve_abs': 'Q',
    'smooth_quadratic_bezier_curveto_rel': 't',
    'smooth_quadratic_bezier_curveto_abs': 'T',
    'elliptical_arc_rel': 'a',
    'elliptical_arc_abs': 'A',
    'close_path': 'z',
}

AttribMapper = {
    'id=': '[<|id=|>]',
    'd=': '[<|d=|>]',
    'fill=': '[<|fill=|>]',
    'stroke-width=': '[<|stroke-width=|>]',
    'stroke-linecap=': '[<|stroke-linecap=|>]',
    'stroke=': '[<|stroke=|>]',
    'opacity=': '[<|opacity=|>]',
    'fill-opacity=': '[<|fill-opacity=|>]',
    'stroke-opacity=': '[<|stroke-opacity=|>]',
    'stroke-dasharray=': '[<|stroke-dasharray=|>]',
    'transform=': '[<|transform=|>]',
    'gradientTransform=': '[<|gradientTransform=|>]',
    'offset=': '[<|offset=|>]',
    'width=': '[<|width=|>]',
    'height=': '[<|height=|>]',
    'cx=': '[<|cx=|>]',
    'cy=': '[<|cy=|>]',
    'rx=': '[<|rx=|>]',
    'ry=': '[<|ry=|>]',
    'r=': '[<|r=|>]',
    'points=': '[<|points=|>]',
    'x1=': '[<|x1=|>]',
    'y1=': '[<|y1=|>]',
    'x2=': '[<|x2=|>]',
    'y2=': '[<|y2=|>]',
    'x=': '[<|x=|>]',
    'y=': '[<|y=|>]',
    'dx=': '[<|dx=|>]',
    'dy=': '[<|dy=|>]',
    'fr=': '[<|fr=|>]',
    'fx=': '[<|fx=|>]',
    'fy=': '[<|fy=|>]',
    'href=': '[<|href=|>]',
    'rotate=': '[<|rotate=|>]',
    'font-size=': '[<|font-size=|>]',
    'font-style=': '[<|font-style=|>]',
    'font-family=': '[<|font-family=|>]',
    'text-anchor=': '[<|text-anchor=|>]',
    'text-content=': '[<|text-content=|>]',
    'preserveAspectRatio=': '[<|preserveAspectRatio=|>]',
    'viewBox=': '[<|viewBox=|>]',
    'class=': '[<|class=|>]',
    'clip-path=': '[<|clip-path=|>]',
    'stop-color=': '[<|stop-color=|>]',
    'stop-opacity=': '[<|stop-opacity=|>]',
    'style=': '[<|style=|>]',
    'font-weight=': '[<|font-weight=|>]',
}

SVGToken = {
    'start': '[<|START_OF_SVG|>]',
    'end': '[<|END_OF_SVG|>]',
}

ContainerMapper = {
        '<svg>': SVGToken['start'],
        '</svg>': SVGToken['end'],
        '<g>': '[<|START_OF_GROUP|>]',
        '</g>': '[<|END_OF_GROUP|>]',
        '<clipPath>': '[<|clipPath|>]',
        '</clipPath>': '[<|/clipPath|>]',
        '<defs>': '[<|defs|>]',
        '</defs>': '[<|/defs|>]',
        '<text>': '[<|text|>]',
        '</text>': '[<|/text|>]',
        '<tspan>': '[<|tspan|>]',
        '</tspan>': '[<|/tspan|>]',
    }

ContainerTagIdentifiers = {
    'svg_start': 'START_OF_SVG',
    'svg_end': 'END_OF_SVG',
    'g_start': 'START_OF_GROUP',
    'g_end': 'END_OF_GROUP',
    'clipPath': 'clipPath',
    'clipPath_end': '/clipPath',
    'defs': 'defs',
    'defs_end': '/defs',
}

PathMapper = {
    '<path>': '[<|path|>]',
    '</path>': '[<|/path|>]'
}

PathIdentifier = 'path'

GradientsMapper = {
    '<linearGradient>': '[<|linearGradient|>]',
    '</linearGradient>': '[<|/linearGradient|>]',
    '<radialGradient>': '[<|radialGradient|>]',
    '</radialGradient>': '[<|/radialGradient|>]',
    '<stop>': '[<|stop|>]',
    '</stop>': '[<|/stop|>]',
}

GradientIdentifier = {
    'linear_gradient': 'linearGradient',
    'radial_gradient': 'radialGradient',
    'stop': 'stop'
}

ShapeMapper = {
    '<circle>': '[<|circle|>]',
    '</circle>': '[<|/circle|>]',
    '<rect>': '[<|rect|>]',
    '</rect>': '[<|/rect|>]',
    '<ellipse>': '[<|ellipse|>]',
    '</ellipse>': '[<|/ellipse|>]',
    '<polygon>': '[<|polygon|>]',
    '</polygon>': '[<|/polygon|>]',
    '<line>': '[<|line|>]',
    '</line>': '[<|/line|>]',
    '<polyline>': '[<|polyline|>]',
    '</polyline>': '[<|/polyline|>]',
    # '<use>': '[<|use|>]',
    # '</use>': '[<|/use|>]',
    '<text>': '[<|text|>]',
    '</text>': '[<|/text|>]',
    '<tspan>': '[<|tspan|>]',
    '</tspan>': '[<|/tspan|>]',
}

ShapeIdentifier = {
    'circle': 'circle',
    'rect': 'rect',
    'ellipse': 'ellipse',
    'polygon': 'polygon',
    'line': 'line',
    'polyline': 'polyline',
    # 'use': 'use',
    'text': 'text',
    'tspan': 'tspan',
}


def remove_square_brackets(s: str) -> str:
    """Remove square brackets from the input string."""
    return s.replace('[', '').replace(']', '')


def is_path_closed(path_d: str) -> bool:
    path_d = path_d.strip()
    return path_d.endswith('Z') or path_d.endswith('z')


def svg2syntactic(
        svg_string: str,
        include_gradient_tag: bool = False,
        path_only: bool = False,
        include_group: bool = True,
        group_attr_inherit: bool = True,
        ignore_tags: List[str] = [],
        ignore_attrs: List[str] = ['gradientUnits'],
) -> Tuple[str, str]:
    tree = etree.fromstring(svg_string)

    struct_ret = ""
    shape_tags = ['circle', 'rect', 'ellipse', 'polygon', 'line', 'polyline', 'use']
    gradient_tags = ['linearGradient', 'radialGradient', 'stop']
    
    # [FIX] 定义允许的标签白名单，防止未定义的标签泄露到 Token 序列中
    ALLOWED_TAGS = set(shape_tags + gradient_tags + ['path', 'text', 'tspan', 'svg', 'g', 'clipPath', 'defs', 'use'])

    stop_attrs = ['offset', 'stop-color']
    gradients = {}
    basic_shape_attrs = [
        'fill', 'stroke-width', 'stroke', 'opacity', 'transform',
        'cx', 'cy', 'r', 'rx', 'ry',
        'width', 'height', 'points',
        'x1', 'y1', 'x2', 'y2',
        'x', 'y', 'dx', 'dy', 'rotate', 'font-size',
        'textLength', 'font-style', 'font-family',
        'fill-opacity', 'stroke-opacity', 'stroke-dasharray',
        'text-anchor', 'preserveAspectRatio', 'viewBox', 'class', 'clip-path',
        'href', 'id'
    ]

    NON_INHERITABLE = {
        'transform', 'clip-path', 'opacity', 'mask', 'filter', 'id', 'class',
        'x', 'y', 'dx', 'dy', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry',
        'x1', 'y1', 'x2', 'y2', 'd', 'points', 'viewBox', 'href'
    }

    def recursive_parse(element, level=0, inherited_attributes=None):
        nonlocal struct_ret
        
        try:
            tag = etree.QName(element).localname
        except Exception as e:
            # print(f"ERROR processing element: {element}, type: {type(element)}")
            # Skip this element if it's invalid (e.g. Comment)
            return

        inherited_attributes = inherited_attributes or {}

        # [FIX] 如果标签不在白名单且不在忽略列表，直接跳过，防止破坏分词
        if tag not in ALLOWED_TAGS and tag not in ignore_tags:
             return

        if tag in ignore_tags:
            return

        # Handle <defs> tag: transparent traversal
        if tag == 'defs':
            struct_ret += "  " * level + f"<{tag}>\n"
            for child in element:
                recursive_parse(child, level + 1, inherited_attributes)
            struct_ret += "  " * level + f"</{tag}>\n"
            return

        # Handle <g> tag: attribute inheritance and optional inclusion
        current_attributes = inherited_attributes.copy()
        if tag == 'g':
            # [FIX] Standardize <g> processing to ensure attributes (transform, etc.) are captured
            # 1. Update inherited attributes for children (Standard SVG inheritance)
            if group_attr_inherit:
                inheritable = {k: v for k, v in element.attrib.items() if k not in NON_INHERITABLE}
                current_attributes.update(inheritable)

            # [FIX] Parse style attributes and update current_attributes for inheritance
            # This ensures that children inherit styles defined in the group's style attribute
            if 'style' in element.attrib:
                style_str = element.attrib['style']
                styles = {}
                for item in style_str.split(';'):
                    if ':' in item:
                        k, v = item.split(':', 1)
                        styles[k.strip().lower()] = v.strip()
                
                # Update current_attributes with styles
                # Only inherit attributes that are typically inheritable
                for key in basic_shape_attrs:
                     if key in styles:
                         current_attributes[key] = styles[key]

            if include_group:
                struct_ret += "  " * level + f"<{tag}>\n"
                
                # [FIX] Parse style attributes for Group as well (for output)
                group_attribs = dict(element.attrib)
                if 'style' in group_attribs:
                    style_str = group_attribs['style']
                    styles = {}
                    for item in style_str.split(';'):
                        if ':' in item:
                            k, v = item.split(':', 1)
                            styles[k.strip().lower()] = v.strip()
                    
                    # Promote style attributes to group attributes if missing
                    for key in basic_shape_attrs:
                         if key in styles and key not in group_attribs:
                             group_attribs[key] = styles[key]

                # Output attributes for the group itself
                attr_str = ""
                for k in basic_shape_attrs:
                    if k in group_attribs and k not in ignore_attrs:
                        v = group_attribs[k]
                        if k == 'transform' and v.startswith('matrix'):
                            v = v.replace(' ', '')
                        attr_str += f" {k}={v}"
                
                if attr_str:
                    struct_ret += "  " * (level + 1) + attr_str.strip() + "\n"

            for child in element:
                recursive_parse(child, level + 1, current_attributes)

            if include_group:
                struct_ret += "  " * level + f"</{tag}>\n"
            return
        
        # Handle <clipPath> tag
        if tag == 'clipPath':
             struct_ret += "  " * level + f"<{tag}>"
             if 'id' in element.attrib:
                 struct_ret += f" id={element.attrib['id']}"
             struct_ret += "\n"
             
             for child in element:
                 recursive_parse(child, level + 1, current_attributes)
             
             struct_ret += "  " * level + f"</{tag}>\n"
             return

        # Get the current element's attributes
        attributes = {**current_attributes, **element.attrib}

        # [FIX] Handle xlink:href -> href mapping for <use> and others
        xlink_href = "{http://www.w3.org/1999/xlink}href"
        if xlink_href in attributes:
             attributes['href'] = attributes[xlink_href]

        # [FIX] Extract style attributes and promote them to top-level attributes
        if 'style' in attributes:
            style_str = attributes['style']
            styles = {}
            for item in style_str.split(';'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    styles[k.strip().lower()] = v.strip()
            
            for key in ['fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity', 'stroke-dasharray', 'font-family', 'font-size', 'font-weight', 'text-anchor']:
                 if key in styles:
                     # If the attribute is NOT explicitly present in the tag's attributes, use the style value.
                     # This allows style to override inherited values, but preserves explicit attributes (e.g. <rect fill="red" style="fill:blue"> -> keeps red if we follow user's hint, though standard is blue).
                     # But crucially, it fixes the "black legend" where `fill` is ONLY in style.
                     if key not in element.attrib:
                        attributes[key] = styles[key]


        # Process <text> or <tspan> tag (Container)
        if tag == "text" or tag == "tspan":
            struct_ret += "  " * level + f"<{tag}>\n"
            
            # [FIX] Update inherited attributes for children (tspan inherits from text)
            # Create a new scope for children
            text_inherited_attributes = current_attributes.copy()
            inheritable = {k: v for k, v in element.attrib.items() if k not in NON_INHERITABLE}
            text_inherited_attributes.update(inheritable)

            # [FIX] Also parse style for inheritance (e.g. <text style="fill:red"><tspan>...)
            if 'style' in element.attrib:
                style_str = element.attrib['style']
                styles = {}
                for item in style_str.split(';'):
                    if ':' in item:
                        k, v = item.split(':', 1)
                        styles[k.strip().lower()] = v.strip()
                
                for key in basic_shape_attrs:
                     if key in styles:
                         text_inherited_attributes[key] = styles[key]

            # Attributes
            number_attrs = {'x', 'y', 'dx', 'dy', 'rotate', 'font-size', 'textLength', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity'}
            
            # Filter and output attributes
            for attr, value in attributes.items():
                if attr in ['x', 'y', 'dx', 'dy', 'rotate', 'font-size', 'textLength', 'font-style', 'text-anchor', 
                            'fill', 'opacity', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray', 
                            'transform', 'class', 'clip-path', 'font-family', 'font-weight', 'style', 'id']:
                    
                    val_str = value
                    if attr in number_attrs:
                         val_str = str(_parse_number(value))
                    elif attr == 'transform':
                         val_str = _parse_transform(value)
                    
                    struct_ret += "  " * (level + 1) + f"{attr}={val_str}\n"

            # Text content
            if element.text and element.text.strip():
                struct_ret += "  " * (level + 1) + f"text-content={element.text.strip()}\n"

            # Children (Recursion)
            for child in element:
                recursive_parse(child, level + 1, text_inherited_attributes)
                
                # [FIX] Handle text after child (tail text)
                if child.tail and child.tail.strip():
                    struct_ret += "  " * (level + 1) + f"text-content={child.tail.strip()}\n"

            struct_ret += "  " * level + f"</{tag}>\n"
            return

        if tag == 'use':
            # Special handling for Natural Language <use>
            # Output raw XML with spaces around '=' to avoid tokenization by _to_svg_description
            attr_str = ""
            for k, v in attributes.items():
                # Add space before = to break regex match for keys like 'x='
                attr_str += f' {k} = "{v}"'
            struct_ret += "  " * level + f"<{tag}{attr_str} />\n"
            return

        # Add current tag to structure (unless ignored tag)
        if tag not in ignore_tags:
            struct_ret += "  " * level + f"<{tag}>\n"

        # Process <path> tag
        if tag == "path" and "d" in attributes:
            struct_ret += "  " * (level + 1) + f"d="
            path_d = attributes['d']
            struct_ret += _parse_path_d(path_d, level + 2, float_coords=True)

            is_closed = is_path_closed(path_d)
            gradient_id = attributes.get('fill')
            if gradient_id and (gradient_id in gradients) and include_gradient_tag:
                struct_ret += "\n" + "  " * (level + 1) + f"{gradients[gradient_id]}\n"
            else:
                struct_ret += f"{_gather_path_attr(attributes)}\n"


        # Process <svg> tag attributes
        elif tag == "svg":
             attr_str = ""
             for attr, value in attributes.items():
                 if attr in ['width', 'height', 'viewBox', 'preserveAspectRatio', 'xmlns', 'version']:
                     if attr in ['width', 'height']:
                         attr_str += f"{attr}={_parse_number(value)} "
                     elif attr == 'viewBox':
                         attr_str += f"{attr}={_parse_point_attr(value, float_coords=True)} "
                     else:
                         attr_str += f"{attr}={value} "
             
             if attr_str:
                 struct_ret += "  " * (level + 1) + attr_str + "\n"

        # Process shape tags
        elif tag in shape_tags and (not path_only):
            point_attrs = {'points'}
            number_attrs = {'cx', 'cy', 'rx', 'ry', 'r', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'dx', 'dy',
                            'rotate', 'font-size', 'textLength', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity'}

            attr_str = ""
            for attr, value in attributes.items():
                if attr in basic_shape_attrs and attr and attr not in ignore_attrs:
                    if attr in point_attrs:
                        attr_str += f"{attr}={_parse_point_attr(value, float_coords=True)} "
                    elif attr in number_attrs:
                        attr_str += f"{attr}={_parse_number(value)} "
                    elif attr == 'transform':
                        attr_str += f"{attr}={_parse_transform(value)} "
                    else:
                        attr_str += f"{attr}={value} "

            struct_ret += "  " * (level + 1) + attr_str + "\n"

        # Process gradient-related tags
        # Process gradient-related tags (preserved in structure)
        elif tag in gradient_tags:
            attr_str = ""
            # Define relevant attributes for gradients and stops
            grad_attrs = ['id', 'x1', 'y1', 'x2', 'y2', 'gradientUnits', 'gradientTransform', 'spreadMethod', 'href',
                          'offset', 'stop-color', 'stop-opacity']
            
            for attr, value in attributes.items():
                if attr in grad_attrs:
                    if attr in ['offset', 'stop-opacity']:
                        attr_str += f"{attr}={_parse_number(value)} "
                    elif attr == 'gradientTransform':
                        attr_str += f"{attr}={_parse_transform(value)} "
                    else:
                        attr_str += f"{attr}={value} "
            
            if attr_str:
                struct_ret += "  " * (level + 1) + attr_str + "\n"

        # [FIX] Disabled special gradient processing to preserve them in structure (e.g. inside defs)
        elif False and tag in gradient_tags:
            gradient_id = attributes.get('id')
            if gradient_id:
                gradient_info = f"<{tag}> " + ' '.join(
                    f"{attr}={'#' + value if attr == 'id' else value}"
                    for attr, value in attributes.items()
                    if attr not in ['gradientUnits', *ignore_attrs]
                )

                xlink_href = attributes.get("{http://www.w3.org/1999/xlink}href")
                if xlink_href:
                    referenced_id = xlink_href.split("#")[-1]
                    if f"url(#{referenced_id})" in gradients:
                        gradients[f"url(#{gradient_id})"] = gradient_info + gradients[f'url(#{referenced_id})']
                else:
                    gradients[f"url(#{gradient_id})"] = gradient_info

                for child in element:
                    if etree.QName(child).localname == 'stop':
                        stop_info = "<stop>" + ' '.join(
                            f"{attr}={value}" for attr, value in child.attrib.items()
                            if attr in stop_attrs
                        ) + " </stop>"
                        gradients[f"url(#{gradient_id})"] += stop_info
                
                gradients[f"url(#{gradient_id})"] += f" </{tag}>"

        for child in element:
            recursive_parse(child, level + 1, current_attributes)

        if tag not in ignore_tags:
            struct_ret += "  " * level + f"</{tag}>\n"

    recursive_parse(tree)
    flatten_struct_ret = struct_ret.replace("\n", "")
    svg_desc_ret = _to_svg_description(flatten_struct_ret)
    return struct_ret, _clean_svg_desc_output(svg_desc_ret)


def _parse_number(num_str: str, round_num: int = 2) -> Union[int, float]:
    try:
        clean_str = num_str.lower().replace('px', '').replace('em', '').replace('%', '')
        num = float(clean_str)
        if num.is_integer():
            return int(num)
        else:
            return round(num, round_num)
    except ValueError:
        return num_str


def _parse_transform(transform_str: str) -> str:
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    def replace_num(match):
        num_str = match.group(0)
        try:
            return str(_parse_number(num_str))
        except ValueError:
            return num_str
    return coord_pattern.sub(replace_num, transform_str)


def _clean_svg_desc_output(svg_string: str):
    svg_string = re.sub(r'\s*\[\s*', '[', svg_string)
    svg_string = re.sub(r'\s*\]\s*', ']', svg_string)
    svg_string = re.sub(r'(\d)\s+', r'\1 ', svg_string)
    svg_string = re.sub(r'\]\s*\[', '][', svg_string)
    return svg_string


def _to_svg_description(input_string: str):
    combined_mapper = {**PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper, **ShapeMapper,
                       **GradientsMapper, **AttribMapper}
    pattern = re.compile('|'.join(map(re.escape, combined_mapper.keys())))

    def replacement(match):
        key = match.group(0)
        if key in combined_mapper:
            return combined_mapper[key]
        return key

    result = pattern.sub(replacement, input_string)
    result = re.sub(r'\]\s+\[', '][', result)
    return result


def _gather_path_attr(path_attributes: str):
    attr_ret = ""
    if path_attributes.get('id', None):
        attr_ret += f" id={path_attributes['id']}"
    if path_attributes.get('fill', None):
        attr_ret += f" fill={path_attributes['fill']}"
    if path_attributes.get('stroke', None):
        attr_ret += f" stroke={path_attributes['stroke']}"
    if path_attributes.get('stroke-linecap', None):
        attr_ret += f" stroke-linecap={path_attributes['stroke-linecap']}"
    if path_attributes.get('stroke-width', None):
        attr_ret += f" stroke-width={_parse_number(path_attributes['stroke-width'])}"
    if path_attributes.get('opacity', None):
        attr_ret += f" opacity={_parse_number(path_attributes['opacity'])}"
    if path_attributes.get('fill-opacity', None):
        attr_ret += f" fill-opacity={_parse_number(path_attributes['fill-opacity'])}"
    if path_attributes.get('stroke-opacity', None):
        attr_ret += f" stroke-opacity={_parse_number(path_attributes['stroke-opacity'])}"
    if path_attributes.get('stroke-dasharray', None):
        attr_ret += f" stroke-dasharray={path_attributes['stroke-dasharray']}"
    if path_attributes.get('clip-path', None):
        attr_ret += f" clip-path={path_attributes['clip-path']}"
    if path_attributes.get('transform', None):
        attr_ret += f" transform={_parse_transform(path_attributes['transform'])}"
    if path_attributes.get('class', None):
        attr_ret += f" class={path_attributes['class']}"
    return attr_ret


def _parse_point_attr(d_string: str, float_coords: bool = False):
    path_d_ret = ""
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    point_values = coord_pattern.findall(d_string)

    if float_coords:
        float_values = [_parse_number(p) for p in point_values if p]
        points = ' '.join(map(str, float_values))
    else:
        points = ' '.join(point_values)
    path_d_ret += f"{points} "
    return path_d_ret


def _parse_path_d(d_string: str, indent_level: int = 0, float_coords: bool = False):
    path_d_ret = ""
    path_command_pattern = re.compile(r'([a-zA-Z])\s*([-0-9.,\s]*)')
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')
    matches = path_command_pattern.findall(d_string)

    for command, points in matches:
        point_values = coord_pattern.findall(points)
        if float_coords:
            float_values = [_parse_number(p) for p in point_values if p]
            points = ' '.join(map(str, float_values))
        else:
            points = ' '.join(point_values)

        if command == 'z':
            path_d_ret += f"[{command}]"
        else:
            path_d_ret += f"[{command}]{points} "
    return path_d_ret


def _extract_elements_and_attributes(svg_description):
    pattern = r"\[([^\]]+)\](.*?)(?=\[|$)"
    matches = re.findall(pattern, svg_description)
    result = [(command.strip(), attributes.strip()) for command, attributes in matches]
    return result


def parse_svg_description(svg_description):
    """
    Fixed parser with robust stack-based state management.
    Handles nested containers, attributes, and leaf nodes correctly.
    """
    svg_params = {'width': '512', 'height': '512', 'viewBox': '0 0 512 512', 'xmlns': 'http://www.w3.org/2000/svg'}
    svg_elements = []
    # Stack items: {'tag': str, 'attrs': dict, 'flushed': bool, 'path_data': list}
    stack = []
    
    # Tag Classifications
    LEAF_TAGS = {'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'path', 'stop', 'use'}
    CONTAINER_TAGS = {'g', 'defs', 'clipPath', 'linearGradient', 'radialGradient', 'text', 'tspan', 'svg'}
    
    # Token to Tag Mapping
    TOKEN_MAP = {}
    for k, v in ShapeIdentifier.items(): TOKEN_MAP[k] = v
    for k, v in GradientIdentifier.items(): TOKEN_MAP[k] = v
    # Fix gradient token mapping (GradientIdentifier uses linear_gradient but token is linearGradient)
    TOKEN_MAP['linearGradient'] = 'linearGradient'
    TOKEN_MAP['radialGradient'] = 'radialGradient'
    
    TOKEN_MAP['START_OF_GROUP'] = 'g'
    TOKEN_MAP['clipPath'] = 'clipPath'
    TOKEN_MAP['defs'] = 'defs'
    TOKEN_MAP['path'] = 'path'
    
    CLOSING_MAP = {
        'END_OF_GROUP': 'g',
        '/clipPath': 'clipPath',
        '/defs': 'defs',
        '/linearGradient': 'linearGradient',
        '/radialGradient': 'radialGradient',
        '/stop': 'stop',
        '/text': 'text',
        '/tspan': 'tspan',
        '/path': 'path'
    }
    # Add closing tags for shapes
    for tag in ShapeIdentifier.keys():
        CLOSING_MAP[f'/{tag}'] = ShapeIdentifier[tag]

    def flush_tag(item, self_closing=False):
        if item['flushed']: return
        tag = item['tag']
        attrs = item['attrs']
        
        # Process path data if present
        if tag == 'path' and item['path_data']:
            d_val = "".join(item['path_data'])
            # If d already in attrs, prepend/append? Usually path tokens build d.
            attrs['d'] = d_val
            
        # Handle text content
        content = attrs.pop('__text_content__', None)
        raw_content = item.get('raw_content')
        
        attr_str = to_attr(attrs)
        
        if self_closing and not content and not raw_content:
            svg_elements.append(f'<{tag} {attr_str} />')
        else:
            svg_elements.append(f'<{tag} {attr_str}>')
            if content:
                svg_elements.append(escape(str(content)))
            if raw_content:
                svg_elements.append(raw_content)
                
        item['flushed'] = True

    tokens = svg_description.split('[<|')
    for i, token in enumerate(tokens):
        if '|>]' not in token: continue
        tag_raw, content = token.split('|>]', 1)
        tag = tag_raw.strip()
        content = content.strip()
        
        if tag.endswith('='):
            tag = tag[:-1]
            
        # 1. Handle SVG Start/End
        if tag in remove_square_brackets(SVGToken['start']):
            svg_elements.append(None) # Placeholder for header
            continue
        elif tag in remove_square_brackets(SVGToken['end']):
            # Close all remaining
            while stack:
                item = stack.pop()
                if not item['flushed']:
                    flush_tag(item, self_closing=False)
                svg_elements.append(f'</{item["tag"]}>')
            svg_elements.append('</svg>')
            break
            
        # 2. Handle Closing Tags
        if tag in CLOSING_MAP:
            target_tag = CLOSING_MAP[tag]
            # Check if stack top matches
            if stack and stack[-1]['tag'] == target_tag:
                item = stack.pop()
                if not item['flushed']:
                    # If it's a leaf and we are closing it explicitly, 
                    # we can treat it as container (open+close) or self-closing.
                    # Standardize on open+close for consistency if content exists.
                    # But for cleaner SVG, use self-closing for empty leaves.
                    is_empty = not item['attrs'].get('__text_content__')
                    flush_tag(item, self_closing=(item['tag'] in LEAF_TAGS and is_empty))
                
                # Only append closing tag if we didn't self-close
                if not (item['tag'] in LEAF_TAGS and is_empty):
                     svg_elements.append(f'</{target_tag}>')
            else:
                # Mismatch or missing closing tags on stack?
                # E.g. stack: [g, rect], token: </g>
                # We should auto-close rect.
                while stack and stack[-1]['tag'] != target_tag:
                    # Auto-close mismatch
                    mismatch = stack.pop()
                    if not mismatch['flushed']:
                         flush_tag(mismatch, self_closing=(mismatch['tag'] in LEAF_TAGS))
                    if mismatch['tag'] not in LEAF_TAGS:
                        svg_elements.append(f'</{mismatch["tag"]}>')
                
                # Now stack top should be target_tag
                if stack and stack[-1]['tag'] == target_tag:
                     item = stack.pop()
                     if not item['flushed']:
                         flush_tag(item, self_closing=False)
                     svg_elements.append(f'</{target_tag}>')
            
            # [FIX] Append any trailing content (e.g. natural language <use> tags)
            if content and content.strip():
                 svg_elements.append(content)

            continue

        # 3. Handle Start Tags
        new_tag = TOKEN_MAP.get(tag)
        if new_tag:
            # Before pushing, check if previous tag on stack needs attention
            if stack:
                parent = stack[-1]
                
                # If parent is LEAF, it cannot contain this new tag. Auto-close parent.
                if parent['tag'] in LEAF_TAGS:
                    stack.pop()
                    if not parent['flushed']:
                        flush_tag(parent, self_closing=True)
                
                # If parent is CONTAINER, flush its opening tag (if not yet) because it has a child now.
                elif not parent['flushed']:
                    flush_tag(parent, self_closing=False)
            
            # Push new tag
            stack.append({'tag': new_tag, 'attrs': {}, 'flushed': False, 'path_data': [], 'raw_content': content})
            continue

        # 4. Handle Attributes & Content
        if stack:
            current = stack[-1]
            if tag == 'text-content':
                # [FIX] If tag is already flushed (e.g. <text> open tag written), append content directly
                if current['flushed']:
                    svg_elements.append(escape(str(content)))
                else:
                    current['attrs']['__text_content__'] = content
            elif tag in PathCMDIdentifier:
                current['path_data'].append(f"{PathCMDIdentifier[tag]}{content}")
            elif tag == 'd':
                # Sometimes d comes as explicit attribute
                pass # handled in loop below? No, content is here.
                # If d is split into tokens, it's handled by PathCMDIdentifier.
                # If d is single token?
                pass
            else:
                # Assume it's an attribute
                current['attrs'][tag] = content
                
            # Special case: width/height/viewBox on stack empty?
            # Original code handled svg params when stack empty.
        elif tag in ['width', 'height', 'viewBox']:
             svg_params[tag] = content

    # Post-loop: Close any remaining stack items
    while stack:
        item = stack.pop()
        if not item['flushed']:
            flush_tag(item, self_closing=(item['tag'] in LEAF_TAGS))
        
        # To be safe: always use </tag> for containers.
        if item['tag'] in CONTAINER_TAGS:
             svg_elements.append(f'</{item["tag"]}>')

    # Construct header
    header = f'<svg height="{svg_params["height"]}" width="{svg_params["width"]}" viewBox="{svg_params["viewBox"]}" xmlns="{svg_params["xmlns"]}">'
    if svg_elements and svg_elements[0] is None:
        svg_elements[0] = header
    elif not svg_elements or not svg_elements[0].startswith('<svg'):
        svg_elements.insert(0, header)
        
    return ''.join(svg_elements)


def to_attr(attributes):
    filtered_attrs = {
        k: v for k, v in attributes.items()
        if v is not None and k != 'text-content' and not k.startswith('_')
    }
    # [FIX] Do not escape quotes if the value is already quoted (for raw text attributes)
    # But wait, to_attr is used for tokenized attributes.
    # Raw attributes in <use> won't pass through here.
    escape_map = {"'": "&apos;", '"': "&quot;"}
    return " ".join(f'{k.replace("_", "-")}="{escape(str(v), escape_map)}"' for k, v in filtered_attrs.items())


def is_next_svg_tag(tokens, i):
    """
    Checks if the next token is the start of an SVG tag.
    [FIX] Added missing closing tags and container tags to prevent parsing errors.
    """
    if i + 1 >= len(tokens):
        return True

    next_token = tokens[i + 1]
    svg_tag_identifiers = set(list(ShapeIdentifier.keys()) +
                              [PathIdentifier] +
                              list(GradientIdentifier.values()) +
                              list(ContainerTagIdentifiers.values()))
    
    # [FIX] Added missing closing tags for gradients and groups
    closing_tags = [f"/{tag}" for tag in ShapeIdentifier.keys()] + \
                   ["/path", "/linearGradient", "/radialGradient", "/stop", "END_OF_GROUP", "END_OF_SVG", "/clipPath", "/defs"]
    svg_tag_identifiers.update(closing_tags)
    
    for identifier in svg_tag_identifiers:
        if next_token.startswith(f"{identifier}|>]"):
            return True

    return False


def syntactic2svg(svg_description: str, print_info: bool = False) -> str:
    if print_info:
        elements_and_attributes = _extract_elements_and_attributes(svg_description)
        print(elements_and_attributes)

    svg_output = parse_svg_description(svg_description)
    return svg_output


# if __name__ == '__main__':
#     # Test case provided
#     string_1f408_200d_2b1b_black_cat = '<svg height="128" viewBox="0 0 128 128" width="128" xmlns="http://www.w3.org/2000/svg"><path d="m37.26 79.78s2.49 8.11-1.2 28.42c-.66 3.65-1.64 7.37-2.13 10.42-6.25 0-6.58 7.12-5.26 7.12h7.45c4.75 0 10.56-11.86 13.7-28.31s-12.56-17.65-12.56-17.65zm46.37 13.17s8.07 4.33 7.78 14.51c-.12 4.02-.89 7.43-1.27 10.75-6.51 0-6.53 7.06-4.64 7.06h6.6c3.28 0 10.67-11.23 12.02-25.89 1.34-14.66-20.49-6.43-20.49-6.43z" fill="#292f33"/><path d="m128 30.03c0-17.5-14.72-26.47-32-26.47-1.97 0-3.56 1.59-3.56 3.55s1.59 3.56 3.56 3.56c6.41 0 23.88 3.32 23.88 19.36 0 10.25-5.57 19.42-13.4 21.27-.58-.26-1.23-.37-1.81-.68-25.74-13.93-47.85 3.21-54.14-1.03-6.23-4.2-1.91-11.57-10.04-18.64-1.17-6.55-3.81-15.92-6.38-15.92-1.95 0-4.5 6.49-6.18 13-2.11-4.91-4.85-9.8-6.62-9.8-2.27 0-4.54 8.08-5.67 15.06-5.97 4.08-10.33 9.98-10.33 16.94 0 9.6 14.4 11.94 19.2 12.13s10.75 12.66 12.75 18.03c4.41 15.53 7.29 30.93 9.59 39.76-5.86 0-6.41 7.51-4.79 7.51 2.53 0 6.94-.01 7.91 0 4.91.05 7.2-16.73 7.2-31.46 0-.76-.04-2.23-.04-2.23s6.87 1.79 21.47-.74c8.69-1.51 17.89 3.02 20.43 11.24 1.88 6.06 4.98 11.75 6.64 15.95-5.65 0-5.49 7.24-3.85 7.24 2.8 0 6.4.05 7.76 0 5.22-.2 2.29-26.93 3.66-35.9 1.38-8.97 4.51-19.83-.81-31.28 11.51-6.31 15.57-19.57 15.57-30.45" fill="#292f33"/><circle cx="21.312" cy="41.841778" fill="#c3c914" r="3.2"/><path d="m10.61 45.72c-2.41 1.51-2.41 6.32-3.61 6.32s-3.61-2.83-3.61-6.32c0-3.48 10.18-1.84 7.22 0m12.78 5.28c-.15.09-.32.13-.51.09-.06-.01-5.5-.86-9.05.52-.45.18-1.21-.08-1.46-.68-.26-.6.08-1.3.54-1.48 5.12-2.02 10.15-.83 10.43-.77.49.1.69.67.67 1.32-.02.41-.37.84-.62 1m-1.68 8.31c-.17.04-.34 0-.5-.11-.05-.03-4.73-2.95-8.54-3.03-.49-.01-1.09-.55-1.09-1.21s.59-1.19 1.08-1.19c5.52.09 9.69 3.17 9.92 3.34.42.29.37.9.1 1.5-.18.38-.67.65-.97.7z" fill="#66757f"/><path d="m27.93 28.02s1.32-.14 3.05.12c.77-1.64 2.08-3.16 2.08-3.16s1 2.38 1.37 4.2c.09.42.78.41 1.15.67 0 0 .13-12.31-3.07-13.64 0-.01-2.69 4-4.58 11.81zm-12.29 5.26s1.37-.95 3.14-2.04c.77-2.23 2.06-4.04 2.06-4.04s1.23 2.26 1.83 4.02c.16.48.86.44 1.2.66 0 0-1.42-12.19-4.54-12.22-.01 0-2.85 4-4.69 11.58z" fill="#66757f"/><circle cx="106.69" cy="41.841778" fill="#c3c914" r="3.2"/><path d="m117.39 45.72c2.41 1.51 2.41 6.32 3.61 6.32s3.61-2.83 3.61-6.32c0-3.48-10.18-1.84-7.22 0m-12.78 5.28c.15.09.32.13.51.09.06-.01 5.5-.86 9.05.52.45.18 1.21-.08 1.46-.68.26-.6-.08-1.3-.54-1.48-5.12-2.02-10.15-.83-10.43-.77-.49.1-.69.67-.67 1.32.02.41.37.84.62 1m1.68 8.31c.17.04.34 0 .5-.11.05-.03 4.73-2.95 8.54-3.03.49-.01 1.09-.55 1.09-1.21s-.59-1.19-1.08-1.19c-5.52.09-9.69 3.17-9.92 3.34-.42.29-.37.9-.1 1.5.18.38.67.65.97.7z" fill="#66757f"/><path d="m100.07 28.02s-1.32-.14-3.05.12c-.77-1.64-2.08-3.16-2.08-3.16s-1 2.38-1.37 4.2c-.09.42-.78.41-1.15.67 0 0-.13-12.31 3.07-13.64 0-.01 2.69 4 4.58 11.81zm12.29 5.26s-1.37-.95-3.14-2.04c-.77-2.23-2.06-4.04-2.06-4.04s-1.23 2.26-1.83 4.02c-.16.48-.86.44-1.2.66 0 0 1.42-12.19 4.54-12.22.01 0 2.85 4 4.69 11.58z" fill="#66757f"/><path d="m50.48 94.22c-5.69 2.53-2.61 5.06-2.61 5.06s-4.3-1.64-5.22-3.8c-1.32-3.09 5.8-2.65 7.83-1.26m-11.23 5.48c.18.31.54.43.83.27.09-.05 7.87-4.32 12.27.53.54.6 1.55.77 2.16.22.6-.55.65-1.48.25-1.93-5.26-5.83-12.26-5.74-12.63-5.7-.76.08-1.31.78-1.22 1.62.04.59.42 1.12.83 1.34m5.27 10.95c.24.11.49.03.65-.18.06-.08 3.55-5.67 8.94-7.95.7-.3 1.25-1.21 1.01-2.03-.23-.83-1.22-1.32-1.92-1.02-7.85 3.33-12.3 9.4-12.55 9.74.43.58-.1 1.35.43 1.94.4.45 1.17.61 1.63.53z" fill="#4a5a65"/><path d="m77.52 94.22c5.69 2.53 2.61 5.06 2.61 5.06s4.3-1.64 5.22-3.8c1.32-3.09-5.8-2.65-7.83-1.26m11.23 5.48c-.18.31-.54.43-.83.27-.09-.05-7.87-4.32-12.27.53-.54.6-1.55.77-2.16.22-.6-.55-.65-1.48-.25-1.93 5.26-5.83 12.26-5.74 12.63-5.7.76.08 1.31.78 1.22 1.62-.04.59-.42 1.12-.83 1.34m-5.27 10.95c-.24.11-.49.03-.65-.18-.06-.08-3.55-5.67-8.94-7.95-.7-.3-1.25-1.21-1.01-2.03.23-.83 1.22-1.32 1.92-1.02 7.85 3.33 12.3 9.4 12.55 9.74.43.58.1 1.35-.43 1.94-.4.45-1.17.61-1.63.53z" fill="#4a5a65"/></svg>'
    
#     print("Testing svg2syntactic...")
#     ret1, ret2 = svg2syntactic(string_1f408_200d_2b1b_black_cat, include_gradient_tag=False, include_group=True)
#     print("Syntactic Structure:\n", ret1)
#     print("Syntactic Description (Token Sequence):\n", ret2)
    
#     print("\nTesting syntactic2svg (Reconstruction)...")
#     ret = syntactic2svg(ret2)
#     print("Reconstructed SVG:\n", ret)