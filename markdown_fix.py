#! /usr/bin/env python3

import os
import sys
import json
import re
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

try:
    import coloredlogs

    coloredlogs.install(level=logging.INFO)
except ImportError:
    pass


# [1] D. G. Lowe, "Distinctive image features from scale-invariant key-points," *Int. J. Comput. Vis.*, vol. 60, no. 2, pp. 91-110, 2004.
REFERENCE_REGEX = re.compile(
    r"^\[(\d+)\]((?:(?!\[\d+\])[^\n])*)\n(?=^\[\d+\]|$)", re.MULTILINE
)

REFERENCE_RANGE_REGEX = re.compile(
    r"\[(\d+)\-(\d+)\]"
)

REFERENCE_SINGLE_REGEX = re.compile(
    r"\[(\d+)\]"
)

class SmartEmbeddedElementReorganizer:
    """智能地将嵌入元素移动到最近的合适标题后面"""
    
    def __init__(self):
        self.heading_patterns = [
            (r'^#{1}\s+(.+)', 1),      # # 一级标题
            (r'^#{2}\s+(.+)', 2),      # ## 二级标题
            (r'^#{3}\s+(.+)', 3),      # ### 三级标题
            (r'^#{4}\s+(.+)', 4),      # #### 四级标题
            (r'^#{5}\s+(.+)', 5),      # ##### 五级标题
            (r'^#{6}\s+(.+)', 6),      # ###### 六级标题
            (r'^([IVX]+)\.\s+(.+)', 1),  # 罗马数字（一级）
            (r'^(\d+)\.\s+([A-Z].+)', 2),  # 1. Title（二级）
            (r'^(\d+\.\d+)\s+(.+)', 3),    # 1.1 Title（三级）
            (r'^([A-Z])\.\s+(.+)', 3),     # A. Title（三级）
            (r'^([a-z])\.\s+(.+)', 4),     # a. Title（四级）
        ]
        
        self.embedded_patterns = {
            'image': r'^!\[.*?\]\(.*?\)$',
            'figure_caption': r'^(Fig\.|Figure|Table|Tab\.)\s+\d+',
            'table_markdown': r'^\|',
            'html_table': r'<table',
        }
    
    def reorganize(self, markdown_text: str) -> str:
        """智能重组文档"""
        lines = markdown_text.split('\n\n')
        
        # 1. 分析文档结构
        doc_elements = self._analyze_document(lines)
        
        # 2. 识别需要移动的嵌入元素
        elements_to_move = self._identify_embedded_elements(doc_elements)
        
        # 3. 重组文档
        reorganized = self._reorganize_elements(doc_elements, elements_to_move)
        
        return '\n\n'.join(reorganized)
    
    def _analyze_document(self, lines: List[str]) -> List[Dict]:
        """分析文档结构，标记每一行的类型和层级"""
        doc_elements = []
        current_heading = None
        current_level = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            element = {
                'index': i,
                'line': line,
                'type': 'text',
                'level': current_level,
                'parent_heading': current_heading,
            }
            
            # 检查是否是标题
            heading_info = self._get_heading_info(line)
            if heading_info:
                element['type'] = 'heading'
                element['level'] = heading_info['level']
                element['heading_text'] = heading_info['text']
                current_heading = i
                current_level = heading_info['level']
                doc_elements.append(element)
                i += 1
            
            # 检查是否是嵌入元素
            elif self._is_embedded_element_start(line):
                embed_lines, end_idx = self._collect_embedded_element(lines, i)
                element['type'] = 'embedded'
                element['lines'] = embed_lines
                element['end_index'] = end_idx - 1
                element['embedded_type'] = self._detect_element_type(embed_lines[0])
                element['nearest_heading'] = current_heading
                element['heading_level'] = current_level
                
                # 检查是否在段落中间
                element['in_paragraph'] = self._is_in_paragraph_context(lines, i, end_idx)
                
                doc_elements.append(element)
                i = end_idx
            
            # 普通文本
            else:
                element['nearest_heading'] = current_heading
                doc_elements.append(element)
                i += 1
        
        return doc_elements
    
    def _get_heading_info(self, line: str) -> Optional[Dict]:
        """获取标题信息"""
        line = line.strip()
        if not line:
            return None
        
        for pattern, level in self.heading_patterns:
            match = re.match(pattern, line)
            if match:
                # 提取标题文本
                if len(match.groups()) >= 2:
                    text = match.group(2)  # 有编号的情况
                elif len(match.groups()) >= 1:
                    text = match.group(1)  # 无编号的情况
                else:
                    text = line
                
                return {
                    'level': level,
                    'text': text,
                    'full_line': line
                }
        
        # 检查其他可能的标题格式
        # 全大写短行
        if len(line) < 80 and line.isupper() and not line.endswith('.'):
            return {'level': 2, 'text': line, 'full_line': line}
        
        # 短行，首字母大写，没有句号（可能是标题）
        if (len(line) < 60 and 
            line[0].isupper() and 
            not line.endswith('.') and
            not re.search(r'\b(is|are|was|were|has|have|will|would)\b', line)):
            return {'level': 3, 'text': line, 'full_line': line}
        
        return None
    
    def _is_embedded_element_start(self, line: str) -> bool:
        """检查是否是嵌入元素的开始"""
        line = line.strip()
        if not line:
            return False
        
        for pattern in self.embedded_patterns.values():
            if re.search(pattern, line):
                return True
        
        return False
    
    def _detect_element_type(self, line: str) -> str:
        """检测嵌入元素类型"""
        line = line.strip()
        
        if re.match(self.embedded_patterns['image'], line):
            return 'image'
        elif re.match(self.embedded_patterns['figure_caption'], line):
            return 'caption'
        elif '|' in line:
            return 'table_markdown'
        elif '<table' in line.lower():
            return 'table_html'
        else:
            return 'unknown'
    
    def _collect_embedded_element(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """收集完整的嵌入元素（包括图片、标题、表格等）"""
        collected = []
        i = start_idx
        element_type = self._detect_element_type(lines[i].strip())
        
        if element_type == 'image':
            # 收集图片及其可能的标题
            collected.append(lines[i])
            i += 1
            
            # 检查后续是否有标题（可能隔空行）
            if i < len(lines):
                if not lines[i].strip():  # 空行
                    if i + 1 < len(lines) and self._is_caption(lines[i + 1]):
                        collected.append(lines[i])  # 保留空行
                        collected.append(lines[i + 1])
                        i += 2
                elif self._is_caption(lines[i]):
                    collected.append(lines[i])
                    i += 1
        
        elif element_type == 'caption':
            # 标题可能在图片/表格前
            collected.append(lines[i])
            i += 1
            
            # 查找相关的图片或表格
            while i < len(lines) and len(collected) < 15:  # 限制搜索范围
                if not lines[i].strip():
                    collected.append(lines[i])
                    i += 1
                elif self._is_embedded_element_start(lines[i]):
                    # 递归收集相关元素
                    sub_elements, sub_end = self._collect_embedded_element(lines, i)
                    collected.extend(sub_elements)
                    i = sub_end
                    break
                else:
                    break
        
        elif element_type == 'table_markdown':
            # Markdown表格
            while i < len(lines):
                line = lines[i].strip()
                # 表格行或分隔符
                if '|' in line or (i == start_idx + 1 and re.match(r'^[\-:\s|]+$', line)):
                    collected.append(lines[i])
                    i += 1
                # 可能的表格标题
                elif self._is_caption(line):
                    collected.append(lines[i])
                    i += 1
                    break
                # 空行可能表示结束
                elif not line and len(collected) > 1:
                    break
                else:
                    break
        
        elif element_type == 'table_html':
            # HTML表格
            table_depth = 0
            while i < len(lines):
                line = lines[i]
                collected.append(line)
                
                table_depth += line.lower().count('<table')
                table_depth -= line.lower().count('</table>')
                
                i += 1
                
                if table_depth == 0:
                    # 检查后续是否有标题
                    if i < len(lines) and self._is_caption(lines[i]):
                        collected.append(lines[i])
                        i += 1
                    break
        
        else:
            # 未知类型
            collected.append(lines[i])
            i += 1
        
        return collected, i
    
    def _is_caption(self, line: str) -> bool:
        """判断是否是图表标题"""
        return bool(re.match(self.embedded_patterns['figure_caption'], line.strip()))
    
    def _is_in_paragraph_context(self, lines: List[str], start_idx: int, end_idx: int) -> bool:
        """判断嵌入元素是否在段落中间"""
        # 检查前一行
        if start_idx > 0:
            prev_line = lines[start_idx - 1].strip()
            if prev_line and not self._get_heading_info(prev_line):
                # 不是标题，且不是完整句子结尾
                if not re.search(r'[.!?。！？]\s*$', prev_line):
                    return True
        
        # 检查后一行
        if end_idx < len(lines):
            next_line = lines[end_idx].strip()
            if next_line and not self._get_heading_info(next_line):
                # 小写开头通常表示句子延续
                if next_line[0].islower():
                    return True
                # 或者前面没有结束，这里继续
                if start_idx > 0 and not re.search(r'[.!?。！？]\s*$', lines[start_idx - 1]):
                    return True
        
        return False
    
    def _identify_embedded_elements(self, doc_elements: List[Dict]) -> List[Dict]:
        """识别需要移动的嵌入元素"""
        elements_to_move = []
        
        for element in doc_elements:
            if element['type'] == 'embedded':
                # 判断应该移动到哪里
                target_heading = self._find_target_heading(doc_elements, element)
                
                if target_heading is not None:
                    element['move_to'] = target_heading
                    element['should_move'] = True
                    elements_to_move.append(element)
                else:
                    element['should_move'] = False
        
        return elements_to_move
    
    def _find_target_heading(self, doc_elements: List[Dict], embed_element: Dict) -> Optional[int]:
        """找到嵌入元素应该移动到的目标标题位置"""
        current_idx = embed_element['index']
        
        # 如果不在段落中间，可能不需要移动
        if not embed_element.get('in_paragraph', False):
            # 检查是否已经紧跟在标题后面
            if embed_element['nearest_heading'] is not None:
                heading_idx = embed_element['nearest_heading']
                # 检查中间是否只有空行
                has_only_empty = True
                for i in range(heading_idx + 1, current_idx):
                    if i < len(doc_elements) and doc_elements[i]['line'].strip():
                        has_only_empty = False
                        break
                
                if has_only_empty:
                    # 已经在合适的位置
                    return None
        
        # 返回最近的标题位置
        return embed_element['nearest_heading']
    
    def _reorganize_elements(self, doc_elements: List[Dict], elements_to_move: List[Dict]) -> List[str]:
        """重组文档元素"""
        result = []
        skip_indices = set()
        
        # 收集所有需要跳过的索引（将被移动的嵌入元素）
        for element in elements_to_move:
            if element['should_move']:
                for i in range(element['index'], element.get('end_index', element['index']) + 1):
                    skip_indices.add(i)
        
        # 按标题分组嵌入元素
        heading_embeds = {}
        for element in elements_to_move:
            if element['should_move'] and element['move_to'] is not None:
                if element['move_to'] not in heading_embeds:
                    heading_embeds[element['move_to']] = []
                heading_embeds[element['move_to']].append(element)
        
        # 重组文档
        i = 0
        while i < len(doc_elements):
            element = doc_elements[i]
            
            # 跳过将被移动的元素
            if i in skip_indices:
                i += 1
                continue
            
            # 添加当前元素
            if element['type'] == 'embedded':
                # 未被移动的嵌入元素
                result.extend(element['lines'])
            else:
                result.append(element['line'])
            
            # 如果是标题，添加属于它的嵌入元素
            if element['type'] == 'heading' and i in heading_embeds:
                # 标题后添加空行
                if result and result[-1].strip():
                    result.append('')
                
                # 按原始顺序添加嵌入元素
                sorted_embeds = sorted(heading_embeds[i], key=lambda x: x['index'])
                
                for embed in sorted_embeds:
                    # 添加嵌入元素
                    result.extend(embed['lines'])
                    # 元素间空行
                    if result and result[-1].strip():
                        result.append('')
            
            i += 1
        
        # 清理多余空行
        return self._clean_result(result)
    
    def _clean_result(self, lines: List[str]) -> List[str]:
        """清理结果，去除多余空行"""
        cleaned = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            
            # 避免连续多个空行
            if is_empty:
                if not prev_empty:
                    cleaned.append(line)
                prev_empty = True
            else:
                cleaned.append(line)
                prev_empty = False
        
        # 去除开头和结尾的空行
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        
        return cleaned

class OCRPageMerger:
    """OCR页面连续性检测与分隔符修复工具"""

    def __init__(self):
        self.metadata_patterns = [
            r"E-mail:",
            r"Manuscript received",
            r"Digital Object Identifier",
            r"Date of publication",
            r"University of",
            r"Department of",
            r"@\w+\.(com|edu|org)",
            r"Recommended for acceptance",
        ]

        self.reference_pattern = r"^\[\d+\]|^\(\d+\)|^\d+\."

        self.embedded_patterns = {
            "image": r"^!\[.*?\]\(.*?\)$",
            "figure_caption": r"^(Fig\.|Figure|Table|Tab\.)\s+\d+",
            "table_markdown": r"^\|?[\s\-:|]+\|",
            "table_html_open": r"<table[^>]*>",
            "table_html_close": r"</table>",
            "table_html_tags": r"<(td|tr|th|thead|tbody)[^>]*>",
        }

    def detect_and_fix(self, markdown_text: str, fix_level: str = "conservative") -> Dict[str, Any]:
        """主修复函数（增强版）"""
        # 替换分隔符格式
        markdown_text = markdown_text.replace("\n---\n", "---")
        pages = markdown_text.split("---")
        
        if len(pages) < 2:
            fixed_text = self.fix_embedded_elements(markdown_text)
            return {
                "fixed_text": fixed_text,
                "changes": [],
                "summary": "单页文档，仅修复嵌入元素",
            }
        
        configs = {
            "conservative": {"min_confidence": 0.8, "strict": True},
            "moderate": {"min_confidence": 0.6, "strict": False},
            "aggressive": {"min_confidence": 0.4, "strict": False},
        }
        config = configs.get(fix_level, configs["conservative"])
        
        result_pages = []
        changes = []
        i = 0
        
        pages = [self.fix_embedded_elements(page) for page in pages]
        
        while i < len(pages):
            current_page = pages[i].strip()
            
            if i < len(pages) - 1:
                next_page = pages[i + 1].strip()
                
                # 使用增强的连续性检测
                continuity_result = self._check_continuity(current_page, next_page)
                
                should_merge = (
                    continuity_result["is_continuous"]
                    and continuity_result["confidence"] >= config["min_confidence"]
                )
                
                if should_merge:
                    # 使用增强的合并方法
                    merged_page = self._merge_pages(
                        current_page, next_page, 
                        continuity_result["fix_type"],
                        continuity_result
                    )
                    result_pages.append(merged_page)
                    
                    changes.append({
                        "action": "merged",
                        "pages": f"{i+1} -> {i+2}",
                        "reason": continuity_result["reason"],
                        "confidence": continuity_result["confidence"],
                        "fix_type": continuity_result["fix_type"],
                        "has_embedded": continuity_result.get("has_embedded", False),
                    })
                    
                    i += 2
                else:
                    result_pages.append(current_page)
                    changes.append({
                        "action": "kept_separate",
                        "pages": f"{i+1} -> {i+2}",
                        "reason": continuity_result["reason"],
                        "confidence": continuity_result["confidence"],
                    })
                    i += 1
            else:
                result_pages.append(current_page)
                i += 1
        
        # 对每个页面再次修复内部的嵌入元素
        result_pages = [self.fix_embedded_elements(page) for page in result_pages]
        
        
        fixed_text = "\n\n".join(result_pages)
        
        fixed_len_L1 = len(fixed_text.split("\n"))
        
        # element_reorg = SmartEmbeddedElementReorganizer()
        # fixed_text = element_reorg.reorganize(fixed_text)
        fixed_len_L2 = len(fixed_text.split("\n"))
        
        
        fixed_text = self.fix_embedded_elements(fixed_text)
        fixed_len_L3 = len(fixed_text.split("\n"))
        
        
        
        merged_count = len([c for c in changes if c["action"] == "merged"])
        kept_count = len([c for c in changes if c["action"] == "kept_separate"])
        
        logging.info(f"fixed_len_L1: {fixed_len_L1}, fixed_len_L2: {fixed_len_L2}, fixed_len_L3: {fixed_len_L3}")
        
        return {
            "fixed_text": fixed_text,
            "changes": changes,
            "summary": f"修复级别: {fix_level}, 合并了 {merged_count} 个页面边界, 保留了 {kept_count} 个分隔符, 处理了嵌入元素",
            "merged_count": merged_count,
            "kept_count": kept_count,
        }
    def _check_continuity(self, current_page: str, next_page: str) -> Dict[str, Any]:
        """检查两页之间的连续性（增强版）"""
        current_lines = [line.strip() for line in current_page.split("\n") if line.strip()]
        next_lines = [line.strip() for line in next_page.split("\n") if line.strip()]
        
        if not current_lines or not next_lines:
            return {
                "is_continuous": False,
                "reason": "页面为空",
                "confidence": 1.0,
                "fix_type": "none",
                "has_embedded": False,
            }
        
        # 检查当前页面末尾是否有嵌入元素
        current_end_idx = len(current_lines) - 1
        embedded_at_end = []
        
        # 向前查找，跳过末尾的嵌入元素
        while current_end_idx >= 0:
            line = current_lines[current_end_idx]
            element_type = self._detect_embedded_element(line)
            
            # 如果是嵌入元素或其标题
            if element_type != 'text' or self._is_figure_caption_simple(line):
                embedded_at_end.insert(0, (current_end_idx, line, element_type))
                current_end_idx -= 1
            else:
                break
        
        # 找到实际的文本结尾
        actual_current_end = current_lines[current_end_idx] if current_end_idx >= 0 else ""
        
        # 检查下一页开头是否有嵌入元素
        next_start_idx = 0
        embedded_at_start = []
        
        while next_start_idx < len(next_lines):
            line = next_lines[next_start_idx]
            element_type = self._detect_embedded_element(line)
            
            if element_type != 'text' or self._is_figure_caption_simple(line):
                embedded_at_start.append((next_start_idx, line, element_type))
                next_start_idx += 1
            else:
                break
        
        # 找到实际的文本开头
        actual_next_start = next_lines[next_start_idx] if next_start_idx < len(next_lines) else ""
        
        # 现在基于实际的文本内容检查连续性
        if actual_current_end and actual_next_start:

            # 1. 检查连字符分割 (最高优先级)
            # 1. 检查连字符分割
            if actual_current_end.endswith("-") and actual_next_start and actual_next_start[0].islower():
                return {
                    "is_continuous": True,
                    "reason": "连字符分割的单词（跨嵌入元素）",
                    "confidence": 0.95,
                    "fix_type": "merge_hyphen_with_embedded",
                    "has_embedded": bool(embedded_at_end or embedded_at_start),
                    "embedded_at_end": embedded_at_end,
                    "embedded_at_start": embedded_at_start,
                }
            
            # 2. 检查句子分割（重要：检查是否是不完整的句子）
            if not self._is_sentence_end(actual_current_end):
                # 特别检查 "we have" + "used" 这种情况
                if actual_next_start and (
                    actual_next_start[0].islower() or 
                    self._looks_like_sentence_continuation(actual_current_end, actual_next_start)
                ):
                    return {
                        "is_continuous": True,
                        "reason": "句子在页面边界被分割（含嵌入元素）",
                        "confidence": 0.85,
                        "fix_type": "merge_sentence_with_embedded",
                        "has_embedded": bool(embedded_at_end or embedded_at_start),
                        "embedded_at_end": embedded_at_end,
                        "embedded_at_start": embedded_at_start,
                    }
        
        current_end = current_lines[-1]
        next_start = next_lines[0]

        # 3. 检查参考文献连续性
        if re.match(self.reference_pattern, current_end.strip()) and re.match(
            self.reference_pattern, next_start.strip()
        ):
            return {
                "is_continuous": True,
                "reason": "参考文献列表连续",
                "confidence": 0.9,
                "fix_type": "merge_references",
            }

        # 4. 检查元数据边界 (这种情况通常不应该合并)
        if self._is_metadata_boundary(current_lines[-3:], next_lines[:3]):
            return {
                "is_continuous": False,
                "reason": "元数据与正文的边界",
                "confidence": 0.9,
                "fix_type": "none",
            }

        # 5. 检查章节边界
        if self._looks_like_heading(next_start) or re.search(
            r"[.!?。！？]\s*$", current_end
        ):
            return {
                "is_continuous": False,
                "reason": "段落或章节边界",
                "confidence": 0.7,
                "fix_type": "none",
            }

        # 6. 检查段落延续
        if len(current_end) > 30 and not current_end.endswith(
            (".", "!", "?", "。", "！", "？", ":", ";")
        ):
            return {
                "is_continuous": True,
                "reason": "段落内容可能被分割",
                "confidence": 0.6,
                "fix_type": "merge_paragraph",
            }

        # 默认：不连续
        return {
            "is_continuous": False,
            "reason": "内容不连续或无法确定",
            "confidence": 0.8,
            "fix_type": "none",
        }

    def _is_metadata_boundary(
        self, current_context: List[str], next_context: List[str]
    ) -> bool:
        """检查是否是元数据边界"""
        current_text = " ".join(current_context)
        next_text = " ".join(next_context)

        current_has_metadata = any(
            re.search(pattern, current_text, re.IGNORECASE)
            for pattern in self.metadata_patterns
        )
        next_has_metadata = any(
            re.search(pattern, next_text, re.IGNORECASE)
            for pattern in self.metadata_patterns
        )

        # 如果当前页有元数据而下页没有，说明是元数据到正文的边界
        return current_has_metadata and not next_has_metadata

    def _looks_like_heading(self, text: str) -> bool:
        """判断是否像标题"""
        return (
            text.startswith("#")  # markdown标题
            or re.match(r"^\d+\.?\s+[A-Z]", text)  # 数字开头的标题
            or (len(text) < 80 and text[0].isupper() and not text.endswith((".")))
        )  # 短且大写开头

    def _looks_like_sentence_continuation(self, prev_text: str, next_text: str) -> bool:
        """判断是否像句子的延续（用于页面边界）"""
        prev_text = prev_text.strip()
        next_text = next_text.strip()
        
        # 检查常见的动词+过去分词模式
        verb_patterns = [
            (r'\b(have|has|had)\s*$', r'^(been|done|made|used|found|shown|given|taken|seen|known)'),
            (r'\b(is|are|was|were|be|being|been)\s*$', r'^(used|made|found|shown|given|based|considered)'),
            (r'\b(will|would|could|should|may|might|must)\s*$', r'^(be|have|use|make|find|show)'),
        ]
        
        for prev_pattern, next_pattern in verb_patterns:
            if re.search(prev_pattern, prev_text, re.IGNORECASE) and \
               re.search(next_pattern, next_text, re.IGNORECASE):
                return True
        
        # 检查介词短语
        if re.search(r'\b(in|on|at|to|from|with|by|for|of)\s*$', prev_text, re.IGNORECASE):
            return True
        
        # 检查连词
        if re.search(r'\b(and|or|but|nor|yet|so)\s*$', prev_text, re.IGNORECASE):
            return True
        
        return False

    def _is_figure_caption_simple(self, text: str) -> bool:
        """简单的图表标题检测（不需要上下文）"""
        caption_pattern = r'^(Fig\.|Figure|Table|Tab\.)\s+\d+'
        return bool(re.match(caption_pattern, text.strip()))
    
    def _merge_pages(self, current_page: str, next_page: str, fix_type: str, 
                     continuity_result: Dict = None) -> str:
        """合并两个页面（处理嵌入元素）"""
        
        if fix_type in ["merge_hyphen_with_embedded", "merge_sentence_with_embedded"]:
            if not continuity_result or not continuity_result.get("has_embedded"):
                # 没有嵌入元素，使用原来的逻辑
                return self._merge_pages_simple(current_page, next_page, fix_type)
            
            current_lines = current_page.split("\n")
            next_lines = next_page.split("\n")
            
            # 获取嵌入元素信息
            embedded_at_end = continuity_result.get("embedded_at_end", [])
            embedded_at_start = continuity_result.get("embedded_at_start", [])
            
            # 找到实际的文本边界
            if embedded_at_end:
                text_end_idx = embedded_at_end[0][0] - 1
            else:
                text_end_idx = len(current_lines) - 1
            
            if embedded_at_start:
                text_start_idx = embedded_at_start[-1][0] + 1
            else:
                text_start_idx = 0
            
            # 构建合并后的内容
            result_lines = []
            
            # 1. 添加当前页面的文本部分（不包括末尾的嵌入元素）
            for i in range(text_end_idx + 1):
                result_lines.append(current_lines[i])
            
            # 2. 合并断开的句子
            if text_end_idx >= 0 and text_start_idx < len(next_lines):
                last_text = result_lines[-1].strip() if result_lines else ""
                next_text = next_lines[text_start_idx].strip()
                
                if fix_type == "merge_hyphen_with_embedded" and last_text.endswith("-"):
                    # 去掉连字符，连接单词
                    result_lines[-1] = last_text[:-1] + next_text
                elif fix_type == "merge_sentence_with_embedded":
                    # 用空格连接句子
                    if last_text and next_text:
                        result_lines[-1] = last_text + " " + next_text
                    elif next_text:
                        result_lines.append(next_text)
                
                # 添加下一页剩余的文本
                for i in range(text_start_idx + 1, len(next_lines)):
                    if not embedded_at_start or i > embedded_at_start[-1][0]:
                        result_lines.append(next_lines[i])
            
            # 3. 添加收集的嵌入元素（放在段落后面）
            if embedded_at_end or embedded_at_start:
                result_lines.append("")  # 空行分隔
                
                # 添加所有嵌入元素
                for _, line, _ in embedded_at_end:
                    result_lines.append(line)
                for _, line, _ in embedded_at_start:
                    result_lines.append(line)
            
            return "\n".join(result_lines)
        
        else:
            # 使用原来的简单合并逻辑
            return self._merge_pages_simple(current_page, next_page, fix_type)
    
    def _merge_pages_simple(self, current_page: str, next_page: str, fix_type: str) -> str:
        """简单的页面合并（原有逻辑）"""
        current_lines = current_page.split("\n")
        next_lines = next_page.split("\n")
        
        if fix_type == "merge_hyphen":
            if current_lines and current_lines[-1].endswith("-"):
                last_line = current_lines[-1].rstrip("-")
                if next_lines:
                    merged_word = last_line + next_lines[0].lstrip()
                    current_lines[-1] = merged_word
                    remaining_lines = next_lines[1:]
                    return "\n".join(current_lines + remaining_lines)
        
        elif fix_type == "merge_sentence":
            if current_lines and next_lines:
                current_lines[-1] += " " + next_lines[0]
                return "\n".join(current_lines + next_lines[1:])
        
        elif fix_type in ["merge_references", "merge_paragraph"]:
            return current_page + "\n" + next_page
        
        return current_page + "\n" + next_page

    def _detect_embedded_element_context(
        self, lines: List[str], idx: int
    ) -> Dict[str, Any]:
        """基于上下文检测嵌入元素类型"""
        if idx >= len(lines):
            return {"type": "text", "is_embedded": False}

        current_line = lines[idx].strip()
        prev_line = lines[idx - 1].strip() if idx > 0 else ""
        next_line = lines[idx + 1].strip() if idx < len(lines) - 1 else ""

        # 检测图片
        if re.match(self.embedded_patterns["image"], current_line):
            return {"type": "image", "is_embedded": True, "has_caption": False}

        # 检测图片标题（需要上下文）
        if self._is_figure_caption_with_context(lines, idx):
            return {"type": "figure_caption", "is_embedded": True}

        # 检测HTML表格
        if self._is_html_table_element(current_line):
            return {"type": "table_html", "is_embedded": True}

        # 检测Markdown表格
        if self._is_markdown_table_element(current_line):
            return {"type": "table_markdown", "is_embedded": True}

        return {"type": "text", "is_embedded": False}

    def _is_figure_caption_with_context(self, lines: List[str], idx: int) -> bool:
        """基于上下文判断是否是图表标题"""
        if idx >= len(lines):
            return False

        current = lines[idx].strip()

        # 检查是否符合标题格式
        caption_pattern = r"^(Fig\.|Figure|Table|Tab\.)\s+\d+"
        if not re.match(caption_pattern, current):
            return False

        # 检查上下文：标题通常在图片前后1-2行内
        # 向前查找图片（标题在图片后）
        for i in range(max(0, idx - 2), idx):
            if re.match(self.embedded_patterns["image"], lines[i].strip()):
                return True
            # 检查是否是HTML图片
            if "<img" in lines[i]:
                return True

        # 向后查找图片（标题在图片前）
        for i in range(idx + 1, min(len(lines), idx + 3)):
            if re.match(self.embedded_patterns["image"], lines[i].strip()):
                return True
            if "<img" in lines[i]:
                return True

        # 检查是否在表格附近
        for i in range(max(0, idx - 2), min(len(lines), idx + 3)):
            if i != idx:
                line = lines[i].strip()
                if self._is_html_table_element(line) or self._is_markdown_table_element(
                    line
                ):
                    return True

        return False

    def _is_html_table_element(self, text: str) -> bool:
        """检测HTML表格元素"""
        html_indicators = [
            "<table",
            "</table>",
            "<tr",
            "</tr>",
            "<td",
            "</td>",
            "<th",
            "</th>",
            "<thead",
            "<tbody",
            "</thead>",
            "</tbody>",
        ]
        return any(indicator in text.lower() for indicator in html_indicators)

    def _is_markdown_table_element(self, text: str) -> bool:
        """检测Markdown表格元素"""
        # Markdown表格特征：包含|且有多个，或包含分隔符
        if "|" in text:
            pipe_count = text.count("|")
            # 至少3个|符号（2列）
            if pipe_count >= 2:
                return True
            # 或者是分隔符行
            if re.match(r"^[\s\-:|]+$", text.replace("|", "")):
                return True
        return False

    def _collect_embedded_element(
        self, lines: List[str], start_idx: int
    ) -> Tuple[List[str], int]:
        """收集完整的嵌入元素（图片+标题或表格）"""
        collected = []
        element_info = self._detect_embedded_element_context(lines, start_idx)
        current_idx = start_idx

        if element_info["type"] == "image":
            # 收集图片
            collected.append(lines[current_idx])
            current_idx += 1

            # 检查后续是否有标题（可能隔一个空行）
            if current_idx < len(lines):
                if not lines[current_idx].strip() and current_idx + 1 < len(lines):
                    # 跳过空行
                    if self._is_figure_caption_with_context(lines, current_idx + 1):
                        collected.append("")  # 保留空行
                        collected.append(lines[current_idx + 1])
                        current_idx += 2
                elif self._is_figure_caption_with_context(lines, current_idx):
                    collected.append(lines[current_idx])
                    current_idx += 1

        elif element_info["type"] == "figure_caption":
            # 标题在前，查找后续图片
            collected.append(lines[current_idx])
            current_idx += 1

            # 可能有空行
            if current_idx < len(lines) and not lines[current_idx].strip():
                collected.append("")
                current_idx += 1

            # 查找图片
            if current_idx < len(lines):
                next_element = self._detect_embedded_element_context(lines, current_idx)
                if next_element["type"] in ["image", "table_html", "table_markdown"]:
                    element_lines, end_idx = self._collect_table_or_image(
                        lines, current_idx
                    )
                    collected.extend(element_lines)
                    current_idx = end_idx

        elif element_info["type"] == "table_html":
            # 收集HTML表格
            table_lines, end_idx = self._collect_html_table(lines, current_idx)
            collected.extend(table_lines)
            current_idx = end_idx

            # 检查表格后是否有标题
            if current_idx < len(lines) and self._is_figure_caption_with_context(
                lines, current_idx
            ):
                collected.append(lines[current_idx])
                current_idx += 1

        elif element_info["type"] == "table_markdown":
            # 收集Markdown表格
            table_lines, end_idx = self._collect_markdown_table(lines, current_idx)
            collected.extend(table_lines)
            current_idx = end_idx

            # 检查表格后是否有标题
            if current_idx < len(lines) and self._is_figure_caption_with_context(
                lines, current_idx
            ):
                collected.append(lines[current_idx])
                current_idx += 1

        return collected, current_idx

    def _collect_html_table(
        self, lines: List[str], start_idx: int
    ) -> Tuple[List[str], int]:
        """收集完整的HTML表格"""
        table_lines = []
        i = start_idx
        table_depth = 0

        while i < len(lines):
            line = lines[i]
            table_lines.append(line)

            # 计算表格标签深度
            table_depth += line.lower().count("<table")
            table_depth -= line.lower().count("</table>")

            i += 1

            # 表格结束
            if table_depth == 0 and "<table" in lines[start_idx].lower():
                break

            # 如果没有明确的结束标签，通过其他方式判断
            if table_depth == 0 and i > start_idx + 1:
                next_line = lines[i].strip() if i < len(lines) else ""
                # 如果下一行不是表格元素，结束收集
                if not self._is_html_table_element(next_line):
                    break

        return table_lines, i

    def _collect_markdown_table(
        self, lines: List[str], start_idx: int
    ) -> Tuple[List[str], int]:
        """收集完整的Markdown表格"""
        table_lines = []
        i = start_idx
        found_separator = False

        while i < len(lines):
            line = lines[i].strip()

            # 空行可能表示表格结束
            if not line and found_separator:
                break

            # 检查是否还是表格行
            if self._is_markdown_table_element(line):
                table_lines.append(lines[i])
                # 检查是否是分隔符行
                if re.match(r"^[\s\-:|]+$", line.replace("|", "")):
                    found_separator = True
                i += 1
            elif table_lines and not line:
                # 空行，可能是表格结束
                break
            elif table_lines:
                # 非表格内容，结束
                break
            else:
                i += 1

        return table_lines, i

    def fix_embedded_elements(self, text: str) -> str:
        """修复被嵌入元素打断的段落"""
        lines = text.split("\n")
        result = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测是否是被打断的段落
            if self._is_incomplete_sentence(line):
                # 收集段落和嵌入元素
                paragraph_parts = []
                embedded_elements = []
                current_part = [lines[i]]
                j = i + 1

                while j < len(lines):
                    element_info = self._detect_embedded_element_context(lines, j)

                    if element_info["is_embedded"]:
                        # 保存当前段落部分
                        if current_part:
                            paragraph_parts.append(" ".join(current_part))
                            current_part = []

                        # 收集嵌入元素
                        element_lines, next_idx = self._collect_embedded_element(
                            lines, j
                        )
                        embedded_elements.append(element_lines)
                        j = next_idx

                    elif self._is_paragraph_continuation(
                        (
                            current_part[-1]
                            if current_part
                            else paragraph_parts[-1] if paragraph_parts else line
                        ),
                        lines[j].strip(),
                    ):
                        # 段落延续
                        current_part.append(lines[j])
                        j += 1

                        # 检查是否句子结束
                        if self._is_sentence_end(lines[j - 1]):
                            # 可能还有更多句子属于同一段落
                            if j < len(lines) and not lines[j].strip():
                                # 遇到空行，段落结束
                                break
                    else:
                        # 不是延续
                        break

                # 保存最后的段落部分
                if current_part:
                    paragraph_parts.append(" ".join(current_part))

                # 重组内容
                if len(paragraph_parts) > 1 or (paragraph_parts and embedded_elements):
                    # 合并段落
                    merged_paragraph = self._merge_paragraph_parts(paragraph_parts)
                    result.append(merged_paragraph)

                    # 添加嵌入元素
                    for element_lines in embedded_elements:
                        result.append("")  # 空行分隔
                        result.extend(element_lines)

                    result.append("")  # 段落后空行
                    i = j
                else:
                    # 没有检测到打断
                    result.append(lines[i])
                    i += 1
            else:
                result.append(lines[i])
                i += 1

        return "\n".join(result)

    def _is_incomplete_sentence(self, text: str) -> bool:
        """检测是否是不完整的句子"""
        if not text:
            return False

        text = text.strip()

        # 空行或太短
        if len(text) < 10:
            return False

        # 检查是否以完整句子结束
        sentence_endings = r"[.!?。！？]\s*$"
        has_ending = bool(re.search(sentence_endings, text))

        # 如果已经有句号结尾，通常是完整的
        if has_ending:
            # 但要检查是否是缩写（如 "Fig." "Dr." "etc."）
            abbreviations = r"\b(Dr|Mr|Ms|Mrs|Prof|Fig|Tab|etc|vs|Inc|Ltd|Co)\.\s*$"
            if re.search(abbreviations, text):
                return True  # 缩写结尾可能不完整
            return False

        # 检查是否包含未完成的语法结构
        incomplete_indicators = [
            r"\b(and|or|but|with|have|has|had|the|a|an|in|on|at|to|from|by|for)\s*$",  # 以连词/介词/冠词结尾
            r",\s*$",  # 以逗号结尾
            r";\s*$",  # 以分号结尾
            r":\s*$",  # 以冒号结尾（可能后面有列表或解释）
            r"\(\s*$",  # 以左括号结尾
            r"-\s*$",  # 以连字符结尾
            r"\s+(is|are|was|were|be|been|being)\s*$",  # 以be动词结尾
        ]

        for pattern in incomplete_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # 检查是否是被截断的长句子（没有句号且较长）
        if len(text) > 50 and not has_ending:
            return True

        return False

    def _is_paragraph_continuation(self, prev_text: str, next_text: str) -> bool:
        """判断下一行是否是段落的延续"""
        if not next_text or not prev_text:
            return False

        next_text = next_text.strip()
        prev_text = prev_text.strip()

        # 跳过空行
        if not next_text:
            return False

        # 检查是否是嵌入元素（不应该作为段落延续）
        if self._detect_embedded_element(next_text) != "text":
            return False

        # 检查是否是标题（新段落开始）
        if self._looks_like_heading(next_text):
            return False

        # 检查是否以小写字母开头（通常是句子延续）
        if next_text[0].islower():
            return True

        # 检查上一行是否以不完整的方式结束
        if self._is_incomplete_sentence(prev_text):
            # 但如果下一行看起来像新段落（如很长的完整句子），则不是延续
            if len(next_text) > 100 and self._is_sentence_end(next_text):
                return False
            return True

        # 检查是否是列表项或编号的延续
        list_patterns = [
            r"^\d+\.",  # 1. 2. 3.
            r"^[a-z]\)",  # a) b) c)
            r"^[•·▪▫◦‣⁃]",  # 项目符号
            r"^[-*+]",  # Markdown列表
        ]

        prev_has_list = any(re.match(p, prev_text) for p in list_patterns)
        next_has_list = any(re.match(p, next_text) for p in list_patterns)

        # 如果都是列表项，可能是延续
        if prev_has_list and next_has_list:
            return False  # 列表项通常是独立的

        # 检查是否是引用的延续
        if prev_text.endswith('"') and '"' in next_text:
            return True

        # 如果上一行很短，下一行很长，可能不是延续
        if len(prev_text) < 30 and len(next_text) > 100:
            return False

        return False

    def _detect_embedded_element(self, text: str) -> str:
        """简单检测文本类型：image/figure/table/text"""
        if not text:
            return "text"

        text = text.strip()

        # 检测图片
        if re.match(self.embedded_patterns["image"], text):
            return "image"

        # 检测图表标题（简单检测，不需要上下文）
        if re.match(self.embedded_patterns["figure_caption"], text):
            return "figure"

        # 检测HTML表格
        if self._is_html_table_element(text):
            return "table_html"

        # 检测Markdown表格
        if self._is_markdown_table_element(text):
            return "table_markdown"

        return "text"

    def _is_sentence_end(self, text: str) -> bool:
        """检查是否是句子结束"""
        text = text.strip()
        if not text:
            return False

        # 句子结束标记
        sentence_endings = r"[.!?。！？]\s*$"
        # 可能的引用结束
        quote_endings = r'[.!?。！？]["\'」』]\s*$'
        # 括号内的句子结束
        paren_endings = r"[.!?。！？]\)\s*$"

        # 检查缩写（这些不算句子结束）
        abbreviations = r"\b(Dr|Mr|Ms|Mrs|Prof|Fig|Tab|etc|vs|Inc|Ltd|Co|Jr|Sr|Ph\.D|M\.D|B\.A|M\.A|D\.D\.S|Ph\.D|U\.S|U\.K|E\.U|i\.e|e\.g)\.\s*$"

        if re.search(abbreviations, text, re.IGNORECASE):
            return False

        return bool(
            re.search(sentence_endings, text)
            or re.search(quote_endings, text)
            or re.search(paren_endings, text)
        )
        
    def _merge_paragraph_parts(self, parts: List[str]) -> str:
        """合并被打断的段落部分"""
        if not parts:
            return ""
        
        merged = []
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            if not merged:
                merged.append(part)
            else:
                prev = merged[-1]
                
                # 如果前一部分以连字符结尾，去掉连字符并直接连接
                if prev.endswith('-'):
                    # 检查是否是真的单词分割
                    if i < len(parts) and parts[i][0].islower():
                        merged[-1] = prev[:-1] + part
                    else:
                        # 可能是破折号，保留
                        merged[-1] = prev + ' ' + part
                # 如果前一部分没有以标点结束，添加空格
                elif not re.search(r'[.!?,;:]\s*$', prev):
                    # 检查是否需要空格
                    if prev and part:
                        # 如果part以标点开始，不加空格
                        if part[0] in '.,;:!?)':
                            merged[-1] = prev + part
                        else:
                            merged[-1] = prev + ' ' + part
                    else:
                        merged.append(part)
                else:
                    # 前面有标点，这是新句子
                    merged[-1] = prev + ' ' + part
        
        return ''.join(merged)
    
    def _collect_table_or_image(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """通用的表格或图片收集方法"""
        element_type = self._detect_embedded_element(lines[start_idx].strip() if start_idx < len(lines) else "")
        
        if element_type == 'image':
            return [lines[start_idx]], start_idx + 1
        elif element_type == 'table_html':
            return self._collect_html_table(lines, start_idx)
        elif element_type == 'table_markdown':
            return self._collect_markdown_table(lines, start_idx)
        else:
            return [lines[start_idx]], start_idx + 1


class MarkdownFixer:
        # 常见“块级”HTML 标签（覆盖 CommonMark/GFM 常见集合）
    BLOCK_HTML_TAGS = (
        "address|article|aside|blockquote|body|caption|center|col|colgroup|dd|details|dialog|div|dl|dt|fieldset|"
        "figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|legend|li|link|main|menu|nav|"
        "noframes|ol|optgroup|option|p|param|section|summary|table|tbody|td|tfoot|th|thead|title|tr|ul|video|audio|canvas"
    )
    # HTML5 空元素（自闭合）
    VOID_HTML_TAGS = "area|base|br|col|embed|hr|img|input|keygen|link|meta|param|source|track|wbr"

    def __init__(self) -> None:
        pass

    def protect(self, text: str) -> str:
        stage1 = self._partition_by_blocks(text)
        stage2 = self._freeze_inline(stage1)
        return stage2

    def unprotect(self, text: str) -> str:
        return text

    @staticmethod
    def _is_blank(line: str) -> bool:
        return len(line.strip()) == 0

    @staticmethod
    def _line_starts_with_fence(line: str) -> Optional[str]:
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        return m.group(1) if m else None

    @staticmethod
    def _line_is_block_math_open(line: str) -> bool:
        return line.strip() == "$$"

    @staticmethod
    def _line_is_block_math_close(line: str) -> bool:
        return line.strip() == "$$"

    @staticmethod
    def _line_starts_html_codey(line: str) -> Optional[str]:
        m = re.search(r"<(pre|code|script|style)(\s|>)", line, flags=re.IGNORECASE)
        return m.group(1).lower() if m else None

    @staticmethod
    def _line_ends_html(tag: str, line: str) -> bool:
        return re.search(rf"</{tag}\s*>", line, flags=re.IGNORECASE) is not None

    @staticmethod
    def _looks_like_table_header(line: str) -> bool:
        # 简化检测：包含至少一个管道符，且不是代码块/HTML 行
        return "|" in line

    @staticmethod
    def _looks_like_table_delim(line: str) -> bool:
        # GFM 风格表头分隔线
        return (
            re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line)
            is not None
        )

    @staticmethod
    def _looks_like_list_item(line: str) -> bool:
        return re.match(r"^\s{0,3}(-|\*|\+|\d{1,9}\.)\s+", line) is not None

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        return re.match(r"^\s{0,3}#{1,6}\s+", line) is not None
    
    @staticmethod
    def _line_starts_block_html_open(line: str) -> str | None:
        """
        若本行以块级 HTML 开始，返回标签名（小写）；若是注释/CDATA/PI，返回特殊标识；否则 None
        """
        s = line.lstrip()
        if s.startswith("<!--"):
            return "__comment__"
        if s.startswith("<![CDATA["):
            return "__cdata__"
        if s.startswith("<?"):
            return "__pi__"
        m = re.match(rf"^<(?P<tag>{MarkdownFixer.BLOCK_HTML_TAGS})\b", s, flags=re.IGNORECASE)
        if m:
            return m.group("tag").lower()
        return None

    @staticmethod
    def _scan_until_html_block_end(lines: List[str], i: int, tag: str) -> int:
        """
        从第 i 行起，扫描到块级 HTML 结束行的下标（包含结束行）。
        - 对 __comment__/__cdata__/__pi__ 做各自的终止符处理
        - 对普通标签做“同名标签计数”处理（支持简单嵌套）
        """
        n = len(lines)
        if tag == "__comment__":
            j = i
            while j < n and "-->" not in lines[j]:
                j += 1
            return min(j, n - 1)
        if tag == "__cdata__":
            j = i
            while j < n and "]]>" not in lines[j]:
                j += 1
            return min(j, n - 1)
        if tag == "__pi__":
            j = i
            while j < n and "?>" not in lines[j]:
                j += 1
            return min(j, n - 1)

        # 空元素：单行即可结束
        if re.match(rf"^(?:{MarkdownFixer.VOID_HTML_TAGS})$", tag, flags=re.IGNORECASE):
            return i

        # 普通块级标签：做简单嵌套计数
        open_pat  = re.compile(rf"<{tag}\b(?![^>]*?/>)", re.IGNORECASE)   # 排除 <tag .../>
        close_pat = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
        depth = 0
        j = i
        while j < n:
            depth += len(open_pat.findall(lines[j]))
            depth -= len(close_pat.findall(lines[j]))
            # 如果第一行就是起始行，确保至少吃到这一行
            if depth <= 0 and j >= i:
                return j
            j += 1
        return n - 1  # 未闭合：吃到文件末尾

    @staticmethod
    def _partition_by_blocks(
        s: str,
    ) -> str:
        """
        块级保护：围栏代码、HTML 代码块、块级数学、表格、脚注定义、缩进代码 等。
        返回替换后的文本。
        """
        lines = s.splitlines(keepends=True)
        i = 0
        out: List[str] = []

        n = len(lines)
        while i < n:
            line = lines[i]

            # 1) 围栏代码 ``` / ~~~
            fence = MarkdownFixer._line_starts_with_fence(line)
            if fence:
                j = i + 1
                while j < n and not re.match(rf"^\s*{re.escape(fence)}", lines[j]):
                    j += 1
                # 包含起止两行
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(block)
                    i = j + 1
                    continue
                else:
                    # 未闭合：按剩余全当代码块处理
                    block = "".join(lines[i:])
                    out.append(block)
                    break

            # 2) HTML 代码型块
            tag = MarkdownFixer._line_starts_html_codey(line)
            if tag:
                j = i
                while j < n and not MarkdownFixer._line_ends_html(tag, lines[j]):
                    j += 1
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(block)
                    i = j + 1
                    continue
                else:
                    block = "".join(lines[i:])
                    out.append(block)
                    break
                
            # 2b) 通用块级 HTML（整块冻结）
            tag_block = MarkdownFixer._line_starts_block_html_open(line)
            if tag_block:
                j = MarkdownFixer._scan_until_html_block_end(lines, i, tag_block)
                block = "".join(lines[i:j+1])
                out.append(block)
                i = j + 1
                continue

            # 3) 块级数学 $$...$$
            if MarkdownFixer._line_is_block_math_open(line):
                j = i + 1
                while j < n and not MarkdownFixer._line_is_block_math_close(lines[j]):
                    j += 1
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(block)
                    i = j + 1
                    continue
                else:
                    block = "".join(lines[i:])
                    out.append(block)
                    break

            # 4) 表格（可配置：整表冻结）
            if (
                i + 1 < n
                and MarkdownFixer._looks_like_table_header(line)
                and MarkdownFixer._looks_like_table_delim(lines[i + 1])
            ):
                j = i + 2
                # 吸收后续表行，直到遇到空行或非表格行
                while (
                    j < n
                    and ("|" in lines[j] or MarkdownFixer._looks_like_table_delim(lines[j]))
                    and not MarkdownFixer._is_blank(lines[j])
                ):
                    j += 1
                block = "".join(lines[i:j])
                out.append(block)
                i = j
                continue

            # 5) 脚注定义
            if re.match(r"^\[\^[^\]]+\]:", line):
                j = i + 1
                # 吸收后续缩进行
                while j < n and (re.match(r"^\s{4,}", lines[j]) or MarkdownFixer._is_blank(lines[j])):
                    j += 1
                block = "".join(lines[i:j])
                out.append(block)
                i = j
                continue

            # 6) 缩进代码块（4 空格起）
            if re.match(r"^( {4}|\t)", line):
                j = i + 1
                while j < n and re.match(r"^( {4}|\t)", lines[j]):
                    j += 1
                block = "".join(lines[i:j])
                out.append(block)
                i = j
                continue

            # 默认：普通行
            out.append(line)
            i += 1

        protected_text =  "\n\n".join(out)
        
        while "\n\n\n" in protected_text:
            protected_text = protected_text.replace("\n\n\n", "\n\n")
        return protected_text
    
    @staticmethod
    def _freeze_inline(text: str) -> str:
        """
        行内保护：图片、链接、行内代码、行内数学、脚注引用、自动链接/内联HTML等。
        注意：按策略控制是否翻译锚文本、图片 alt。
        """
        s = text

        # 0) 链接定义（形如 [id]: http...），若块级没吸到，这里兜底
        # def repl_link_def(m: re.Match) -> str:
        #     return m.group(0)

        # s = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", repl_link_def, s, flags=re.MULTILINE)

        # # 1) 图片 ![alt](url "title") —— 默认整体冻结；若允许翻 alt，则改为仅冻结 ()，放开 []
        # img_pattern = re.compile(r"!\[(?:[^\]\\]|\\.)*?\]\((?:[^()\\]|\\.)*?\)")
        # s = img_pattern.sub(lambda m: m.group(0), s)


        # # 2) 普通链接 [text](url) —— 默认整体冻结；若开启，则仅冻结 (url)，放开 [text]
        # link_pattern = re.compile(r"\[(?:[^\]\\]|\\.)*?\]\((?:[^()\\]|\\.)*?\)")
        # s = link_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        # # 3) 引用式链接 [text][id] —— 整体冻结（或按策略仅放开 [text]）
        # ref_link_pattern = re.compile(r"\[(?:[^\]\\]|\\.)*?\]\[[^\]]+\]")
        # s = ref_link_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        # # 4) 自动链接 <http...> / <mailto:...> —— 冻结
        # autolink_pattern = re.compile(r"<(?:https?://|mailto:)[^>]+>")
        # s = autolink_pattern.sub(lambda m: " " + m.group(0) + " ", s)
        
        # 4b) URL, 邮箱, 电话号码
        # 规则说明（URL）：
        #  - (?<!<)         ：左侧不是 '<'（已被 <...> 包裹时跳过）
        #  - (?<!]\()       ：左侧不是 "]("（跳过 [文本](URL) 里的 URL）
        #  - (?:(?<=^)|(?<=\s)|(?<=[\(\[{\"“])) ：左边界是行首/空白/常见左括引号（避免吃掉左侧字符）
        #  - (https?://[^\s\)\]\}>]+)           ：URL 本体，排除空白与常见右侧闭合符
        #  - (?=[\s\)\]\}>.,!?;:，。！？；：]|$) ：右侧必须是空白/闭合符/标点/行尾
        url_pattern = re.compile(
            r'(?<!<)(?<!]\()(?:(?<=^)|(?<=\s)|(?<=[\(\[{\"“]))'
            r'(https?://[^\s\)\]\}>]+)'
            r'(?=[\s\)\]\}>.,!?;:，。！？；：]|$)'
        )

        def bracket_urls(s: str) -> str:
            return url_pattern.sub(lambda m: f'<{m.group(1)}>', s)
        # 规则说明（邮箱）：
        #  - (?<!<)               ：左侧不是 '<'（已被 <...> 包裹时跳过）
        #  - (?<!]\()             ：左侧不是 "]("（跳过 [文本](mailto:...) 的“万一”误伤场景）
        #  - (?<![\w.@]) (…) (?![\w.@]) ：左右不是邮箱可组成字符，避免粘连
        # 左侧排除：已用 <...> 包裹、Markdown 的 ](…)
        # 左侧再用 (?<![\w.%+-]) 防止从单词中间起跳
        # 邮箱本体：捕获到 \1
        # 右侧用正向前瞻，允许为空白/闭合符/常见中英文标点/行尾，但不吃进去
        email_pattern = re.compile(
            r'(?<!<)(?<!]\()(?<![\w.%+-])'
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
            r'(?=[\s\)\]\}>.,!?;:，。！？；：]|$)'
        )

        def bracket_emails(s: str) -> str:
            return email_pattern.sub(r'<mailto:\1>', s)
        
        s = bracket_urls(s)
        s = bracket_emails(s)

        # 5) 行内代码 `...`（反引号可变宽度）
        inline_code_pattern = re.compile(r"(?<!`)(`+)([^`\n]+?)\1(?!`)")
        s = inline_code_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        # 6) 行内数学 $...$（简化版）
        inline_math_pattern = re.compile(r"\$(?!\s)([^$\n]+?)\$(?!\$)")
        s = inline_math_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        # # 7) 脚注引用 [^id]
        # footref_pattern = re.compile(r"\[\^[^\]]+\]")
        # s = footref_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        # 8) 内联 HTML（保守起见冻结；若想翻译其中文本，可做白名单标签）
        # inline_html_pattern = re.compile(
        #     r"<[A-Za-z][^>]*?>.*?</[A-Za-z][^>]*?>", re.DOTALL
        # )
        # s = inline_html_pattern.sub(lambda m: " " + m.group(0) + " ", s)

        return s


def fix_reference(markdown_content: str):
    # use regex to find the reference

    all_references = re.findall(REFERENCE_REGEX, markdown_content)

    logging.info(f"Found {len(all_references)} references")
    # add \n to every reference
    for match in all_references:
        original_reference_text = f"[{match[0]}] {match[1].strip()}"
        # print(reference_text)
        reference_text = f"[^{match[0]}]: {match[1].strip()}"
        markdown_content = markdown_content.replace(
            original_reference_text, f"{reference_text}\n"
        )
    
    all_reference_ranges = re.findall(REFERENCE_RANGE_REGEX, markdown_content)
    for match in all_reference_ranges:
        original_reference_text = f"[{match[0]}-{match[1]}]"
        range_list = list(range(int(match[0]), int(match[1]) + 1))
        reference_text = "".join([f"[{i}]" for i in range_list])
        markdown_content = markdown_content.replace(
            original_reference_text, f"{reference_text}"
        )
        
    all_reference_items = re.findall(REFERENCE_SINGLE_REGEX, markdown_content)
    for match in all_reference_items:
        original_reference_text = f"[{match}]"
        reference_text = f"[^{match}]"
        markdown_content = markdown_content.replace(
            original_reference_text, f"{reference_text}"
        )
    
    return markdown_content

def fix_complex_titles(markdown_content: str):
    """
    修复各种格式的标题，包括罗马数字、字母编号等
    """
    
    # 定义不同格式的标题模式和对应的级别
    title_patterns = [
        # (正则表达式, 级别, 描述)
        (r'^(#{1,6})\s*(Sec(?:tion)?\.?\s+)?([IVX]+)\.?\s+(.+)$', 2, 'roman_with_sec'),  # Sec V. Title
        (r'^(#{1,6})\s*([IVX]+)\.?\s+(.+)$', 2, 'roman'),  # V. Title
        (r'^(#{1,6})\s*(\d+)\.?\s+([A-Z].+)$', 2, 'number'),  # 1. Title
        (r'^(#{1,6})\s*(\d+\.\d+)\.?\s+(.+)$', 3, 'subsection'),  # 1.1 Title
        (r'^(#{1,6})\s*(\d+\.\d+\.\d+)\.?\s+(.+)$', 4, 'subsubsection'),  # 1.1.1 Title
        (r'^(#{1,6})\s*([A-Z])\.?\s+(.+)$', 3, 'upper_letter'),  # A. Title
        (r'^(#{1,6})\s*([a-z])\.?\s+(.+)$', 4, 'lower_letter'),  # a. Title
    ]
    
    # 先扫描文档，确定是否存在罗马数字标题（判断文档结构类型）
    has_roman = bool(re.search(r'^#{1,6}?\s*(?:Sec(?:tion)?\.?\s+)?[IVX]+\.?\s+', markdown_content, re.MULTILINE))
    
    lines = markdown_content.split('\n')
    new_lines = []
    
    for line in lines:
        modified = False
        
        # 处理带 Sec 的罗马数字标题
        if has_roman:
            match = re.match(r'^(#{1,6})?\s*(Sec(?:tion)?\.?\s+)?([IVX]+)\.?\s+(.+)$', line)
            if match:
                current_hashes = match.group(1) or ''
                section_prefix = match.group(2) or ''
                roman_num = match.group(3)
                title = match.group(4)
                
                # 罗马数字作为一级标题
                new_hashes = '#'
                new_line = f"{new_hashes} {section_prefix}{roman_num}. {title}"
                new_lines.append(new_line)
                modified = True
                logging.debug(f"Roman: '{line}' -> '{new_line}'")
        
        if not modified:
            # 处理数字编号（可能是多级）
            match = re.match(r'^(#{1,6})?\s*(\d+(?:\.\d+)*?)\.?\s+(.+)$', line)
            if match:
                current_hashes = match.group(1) or ''
                number = match.group(2)
                title = match.group(3)
                
                # 根据点的数量确定级别
                level = len(number.split('.'))
                if has_roman:
                    level += 1  # 如果有罗马数字，所有数字编号降一级
                
                new_hashes = '#' * min(level, 6)  # 最多6级
                new_line = f"{new_hashes} {number}. {title}"
                new_lines.append(new_line)
                modified = True
                logging.debug(f"Number: '{line}' -> '{new_line}'")
        
        if not modified:
            # 处理大写字母编号
            match = re.match(r'^(#{1,6})?\s*([A-Z])\.?\s+(.+)$', line)
            if match and not re.match(r'^[A-Z][a-z]', match.group(3)):  # 确保不是普通句子
                current_hashes = match.group(1) or ''
                letter = match.group(2)
                title = match.group(3)
                
                # 字母编号通常是三级标题
                level = 3
                if has_roman:
                    level = 3  # 在有罗马数字的文档中保持三级
                
                new_hashes = '#' * level
                new_line = f"{new_hashes} {letter}. {title}"
                new_lines.append(new_line)
                modified = True
                logging.debug(f"Letter: '{line}' -> '{new_line}'")
        
        if not modified:
            # 处理小写字母编号
            match = re.match(r'^(#{1,6})?\s*([a-z])\.?\s+(.+)$', line)
            if match:
                current_hashes = match.group(1) or ''
                letter = match.group(2)
                title = match.group(3)
                
                # 小写字母通常是四级标题
                level = 4
                if has_roman:
                    level = 4
                
                new_hashes = '#' * level
                new_line = f"{new_hashes} {letter}. {title}"
                new_lines.append(new_line)
                modified = True
                logging.debug(f"Lower letter: '{line}' -> '{new_line}'")
        
        if not modified:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

ALG_HEADER_RE = re.compile(
    r'^\s*(\*\*)?\s*(Algorithm|算法)\s+([A-Za-z0-9.-]+)?\s*(\*\*)?\s*(.*)$',
    re.IGNORECASE
)

BOLD_LINE_RE = re.compile(r'^\s*\*\*.+\*\*\s*$')

# 允许继续属于算法块的行（启发式）
def _is_algo_continuation(line: str) -> bool:
    s = line.strip()
    if s == "":
        return True
    if s == "***":
        return True
    # 标题化的键
    if re.match(r'^\s*\*\*\s*(Input|Inputs|Output|Outputs|Procedure|Requires?|Ensure|Precondition|Postcondition)\s*:?\s*\*\*', s, re.I):
        return True
    if re.match(r'^\s*(Input|Inputs|Output|Outputs|Procedure|Requires?|Ensure|Precondition|Postcondition)\s*:?', s, re.I):
        return True
    # 编号步骤：1: / 1. / 1)
    if re.match(r'^\s*\d+\s*[:.)]\s', s):
        return True
    # 常见伪代码关键词开头
    if re.match(r'^\s*(function|procedure|for|while|if|else|repeat|return|end)\b', s, re.I):
        return True
    return False

def _clean_inline(text: str) -> str:
    # <sub>…</sub> -> _…
    text = re.sub(r'<\s*sub\s*>\s*(.*?)\s*<\s*/\s*sub\s*>', lambda m: '_' + re.sub(r'\*', '', m.group(1)), text, flags=re.I)
    # 去掉**加粗** 与 *斜体*
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    # 把多余空白压一压（不动缩进）
    return text

def _pseudocode_format_fix(text: str) -> str:
    """
    伪代码格式化
    """
    text = text.replace("&nbsp;", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    return text

def remove_unnecessary_whitespaces(text: str) -> str:
    """
    移除不必要的空格
    """
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text

def wrap_pseudocode_blocks(md: str, lang: str = "pseudo") -> str:
    """
    识别以 'Algorithm/算法' 开头的伪代码块，并包裹为 ```pseudo fenced block。
    内部做少量清洗：去粗斜体、<sub>…</sub>→_…、把分隔线 *** 去掉。
    已在 ``` 代码块内的区域不会再次处理。
    """
    lines = md.splitlines()
    out = []
    i = 0
    in_fence = False

    while i < len(lines):
        line = lines[i]

        # 维护已有代码块围栏，避免重复包裹
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if (not in_fence) and ALG_HEADER_RE.match(line):
            # 捕获算法块
            header_line = line
            block = [header_line]
            i += 1
            saw_step = False

            # 向下收集：直到遇到明显的新段/标题/代码围栏/非算法行
            while i < len(lines):
                peek = lines[i]
                if peek.strip().startswith("```"):
                    break
                # markdown 标题
                if re.match(r'^\s*#{1,6}\s', peek):
                    break
                # 明显新粗体标题（但排除 **Input:** 这类）
                if BOLD_LINE_RE.match(peek) and not re.match(r'^\s*\*\*\s*(Input|Output|Procedure|Requires?|Ensure)\s*[:：]?\s*\*\*', peek, re.I):
                    break

                if not _is_algo_continuation(peek):
                    # 若已经收集到步骤，再遇到非算法行就收尾
                    if saw_step:
                        break
                    # 若还没开始步骤，允许吃一两行上下文（如 Input/Procedure）
                    # 这里仍旧放行一行，然后继续判断下一行
                    block.append(peek)
                    i += 1
                    continue

                if re.match(r'^\s*\d+\s*[:.)]\s', peek.strip()):
                    saw_step = True

                block.append(peek)
                i += 1

            # 规范化并写出 fenced block
            norm = []
            # 处理标题行 → 注释化
            m = ALG_HEADER_RE.match(header_line)
            if m:
                alg_no = (m.group(3) or "").strip()
                rest = (m.group(5) or "").strip()
                title = f"Algorithm {alg_no}: {rest}".strip() if alg_no or rest else "Algorithm"
                norm.append(f"// { _clean_inline(title) }")
            else:
                norm.append(f"// { _clean_inline(header_line) }")

            for raw in block[1:]:
                s = raw.strip()
                if s == "***":
                    # 三星分隔线转注释横线
                    norm.append("// " + "-" * 40)
                    continue
                norm.append(_clean_inline(raw))

            out.append(f"```{lang}")
            out.extend(norm)
            out.append("```")
            continue

        # 普通行照抄
        out.append(line)
        i += 1


    
    return "\n".join(out)

def fix_titles(markdown_content: str):
    """
    智能版本：自动检测文档结构并调整标题级别
    """
    
    class TitleInfo:
        def __init__(self, line_num, original, type_, number, title):
            self.line_num = line_num
            self.original = original
            self.type = type_
            self.number = number
            self.title = title
            self.level = None
    
    # 收集所有标题
    titles = []
    lines = markdown_content.split('\n')
    
    for i, line in enumerate(lines):
        # 罗马数字（可能带 Sec）
        match = re.match(r'^(#{1,6})+\s*(?:Sec(?:tion)?\.?\s+)?([IVX]+)\.?\s+(.+)$', line)
        if match:
            titles.append(TitleInfo(i, line, 'roman', match.group(2), match.group(3)))
            continue
        
        # 数字编号（支持多级）
        match = re.match(r'^(#{1,6})+\s*(\d+(?:\.\d+)*?)\.?\s+(.+)$', line)
        if match:
            titles.append(TitleInfo(i, line, 'number', match.group(2), match.group(3)))
            continue
        
        # 大写字母
        match = re.match(r'^(#{1,6})+\s*([A-Z])\.?\s+([A-Z].+)$', line)
        if match:
            titles.append(TitleInfo(i, line, 'upper', match.group(2), match.group(3)))
            continue
        
        # 小写字母
        match = re.match(r'^(#{1,6})+\s*([a-z])\.?\s+(.+)$', line)
        if match:
            titles.append(TitleInfo(i, line, 'lower', match.group(2), match.group(3)))
    
    # 分析层级结构
    has_roman = any(t.type == 'roman' for t in titles)
    
    # 分配级别
    for title in titles:
        if title.type == 'roman':
            title.level = 2
        elif title.type == 'number':
            dots = title.number.count('.')
            base_level = 3 if has_roman else 2
            title.level = base_level + dots
        elif title.type == 'upper':
            title.level = 3
        elif title.type == 'lower':
            title.level = 4
        
        # 确保不超过6级
        title.level = min(title.level, 6)
    
    # 重建内容
    title_dict = {t.line_num: t for t in titles}
    new_lines = []
    
    for i, line in enumerate(lines):
        if i in title_dict:
            t = title_dict[i]
            hashes = '#' * t.level
            
            if t.type == 'roman':
                # 保留 Sec 前缀（如果有）
                if 'Sec' in t.original:
                    new_line = f"{hashes} Sec {t.number}. {t.title}"
                else:
                    new_line = f"{hashes} {t.number}. {t.title}"
            else:
                new_line = f"{hashes} {t.number}. {t.title}"
            
            new_lines.append(new_line)
            logging.debug(f"Fixed: '{line}' -> '{new_line}'")
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_content(markdown_content: str) -> str:
    # fixed_content = fix_reference(markdown_content)
    fixed_content = fix_titles(markdown_content)
    
    continue_fixer = OCRPageMerger()

    result = continue_fixer.detect_and_fix(fixed_content, level)
    

    logging.info(
        f"Fixed {result['merged_count']} page boundaries, kept {result['kept_count']} page boundaries"
    )

    fixed_content = fix_reference(result["fixed_text"])
    
    fixed_content = wrap_pseudocode_blocks(fixed_content)
    
    fixed_content = _pseudocode_format_fix(fixed_content)
    
    fixer = MarkdownFixer()
    fixed_content = fixer.protect(fixed_content)
    
    fixed_content = remove_unnecessary_whitespaces(fixed_content)
    
    return fixed_content
    
def fix_file(markdown_file: str, output_file: str, level: str = "moderate"):
    if not os.path.exists(markdown_file):
        logging.error(f"File {markdown_file} does not exist")
        return

    logging.info(f"Fixing file {markdown_file} with level {level}")

    output_dir = os.path.abspath(os.path.dirname(output_file))
    os.makedirs(output_dir, exist_ok=True)

    with open(markdown_file, "r") as f:
        markdown_content = f.read()

    fixed_content = fix_content(markdown_content)

    with open(output_file, "w") as f:
        f.write(fixed_content)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Fix markdown file")
    parser.add_argument("markdown_file", type=str, help="Markdown file to fix")
    parser.add_argument("-o", "--output_file", type=str, help="Output file")
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=2,
        help="Fix level: \n 1. conservative: only merge if the confidence is high \n 2. moderate: merge if the confidence is medium \n 3. aggressive: merge if the confidence is low",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "-i",
        "--inplace",
        action="store_true",
        help="Inplace fixed",
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    level = ""
    if args.level == 1:
        level = "conservative"
    elif args.level == 2:
        level = "moderate"
    elif args.level == 3:
        level = "aggressive"

    if args.inplace:
        fix_file(args.markdown_file, args.markdown_file, level)
    else:
        fix_file(args.markdown_file, args.output_file, level)
