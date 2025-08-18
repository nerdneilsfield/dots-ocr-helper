#! /usr/bin/env python3
"""
Markdown修复工具 - Linus风格重构版

原则：
1. 好品味：用正确的数据结构消除特殊情况
2. 简洁：如果超过3层缩进就重新设计
3. 实用：解决真实问题，不搞过度设计
"""

import os
import sys
import json
import re
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    import coloredlogs
    coloredlogs.install(level=logging.INFO)
except ImportError:
    pass


# ============================================================================
# 数据结构：一切的基础
# ============================================================================

class BlockType(Enum):
    """块类型 - 用枚举替代字符串比较"""
    TEXT = auto()
    HEADING = auto()
    CODE_FENCE = auto()
    MATH_BLOCK = auto()
    TABLE = auto()
    IMAGE = auto()
    FIGURE_CAPTION = auto()
    LIST_ITEM = auto()
    REFERENCE = auto()
    HTML_BLOCK = auto()
    METADATA = auto()
    PAGE_BOUNDARY = auto()


@dataclass
class DocumentBlock:
    """文档块 - 统一表示所有内容"""
    type: BlockType
    content: str
    line_start: int
    line_end: int
    page_num: Optional[int] = None
    confidence: float = 1.0
    source: str = "original"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_incomplete_sentence(self) -> bool:
        """检查是否是不完整的句子"""
        if self.type != BlockType.TEXT:
            return False
        text = self.content.strip()
        if len(text) < 10:
            return False
        # 简单规则：没有句号结尾且超过50字符
        return len(text) > 50 and not re.search(r'[.!?。！？]\s*$', text)
    
    def can_merge_with(self, other: 'DocumentBlock') -> Tuple[bool, str]:
        """检查是否能与另一个块合并"""
        if not isinstance(other, DocumentBlock):
            return False, "不是DocumentBlock"
        
        # 页面边界不合并
        if self.type == BlockType.PAGE_BOUNDARY or other.type == BlockType.PAGE_BOUNDARY:
            return False, "页面边界"
        
        # 不同类型一般不合并  
        if self.type != other.type:
            return False, f"类型不同: {self.type} vs {other.type}"
        
        # 文本块的合并逻辑
        if self.type == BlockType.TEXT:
            # 检查连字符分割
            if self.content.strip().endswith('-') and other.content.strip() and other.content.strip()[0].islower():
                return True, "连字符分割"
            # 检查句子延续
            if self.is_incomplete_sentence():
                return True, "句子不完整"
        
        return False, "默认不合并"


@dataclass 
class Document:
    """文档类 - 管理所有块"""
    blocks: List[DocumentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_block(self, block: DocumentBlock) -> None:
        """添加块"""
        self.blocks.append(block)
    
    def merge_consecutive_blocks(self) -> int:
        """合并连续的块"""
        if len(self.blocks) < 2:
            return 0
        
        merged_count = 0
        new_blocks = []
        i = 0
        
        while i < len(self.blocks):
            current = self.blocks[i]
            
            # 看看能不能和下一个块合并
            if i + 1 < len(self.blocks):
                next_block = self.blocks[i + 1]
                can_merge, reason = current.can_merge_with(next_block)
                
                if can_merge:
                    # 合并内容
                    if reason == "连字符分割":
                        merged_content = current.content.rstrip('-') + next_block.content.lstrip()
                    else:
                        merged_content = current.content.rstrip() + ' ' + next_block.content.lstrip()
                    
                    # 创建新的合并块
                    merged_block = DocumentBlock(
                        type=current.type,
                        content=merged_content,
                        line_start=current.line_start,
                        line_end=next_block.line_end,
                        source="merged",
                        metadata={'merge_reason': reason}
                    )
                    
                    new_blocks.append(merged_block)
                    merged_count += 1
                    i += 2  # 跳过下一个块
                    continue
            
            # 不能合并，直接添加
            new_blocks.append(current)
            i += 1
        
        self.blocks = new_blocks
        return merged_count
    
    def to_markdown(self) -> str:
        """转换回Markdown"""
        lines = []
        for block in self.blocks:
            if block.type == BlockType.PAGE_BOUNDARY:
                continue  # 跳过页面边界标记
            lines.append(block.content)
        
        result = '\n'.join(lines)
        
        # 清理多余空行
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        
        return result


# ============================================================================
# 解析器：字符串 → 结构化数据
# ============================================================================

class DocumentParser:
    """文档解析器 - 统一解析逻辑"""
    
    def __init__(self):
        # 预编译正则表达式，提高性能
        self.patterns = {
            'heading': re.compile(r'^(#{1,6})\s+(.+)'),
            'code_fence': re.compile(r'^\s*(`{3,}|~{3,})'),
            'image': re.compile(r'^!\[.*?\]\(.*?\)$'),
            'figure_caption': re.compile(r'^(Fig\.|Figure|Table|Tab\.)\s+\d+'),
            'table_markdown': re.compile(r'^\|'),
            'reference': re.compile(r'^\[(\d+)\]'),
            'page_boundary': re.compile(r'^---$'),
        }
    
    def parse(self, text: str) -> Document:
        """解析文本为文档对象"""
        # 精确分割页面：只有独立成行的 --- 才是页面分隔符
        pages = re.split(r'\n---\n', text)
        
        doc = Document()
        line_number = 0
        
        for page_num, page_content in enumerate(pages):
            if page_num > 0:
                # 添加页面边界标记
                doc.add_block(DocumentBlock(
                    type=BlockType.PAGE_BOUNDARY,
                    content='---',
                    line_start=line_number,
                    line_end=line_number,
                    page_num=page_num
                ))
                line_number += 1
            
            # 解析页面内容
            lines = page_content.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i]
                start_line = line_number
                
                # 检测块类型
                block_type, end_i = self._detect_block_type(lines, i)
                
                # 提取内容
                content = '\n'.join(lines[i:end_i+1])
                
                # 跳过纯空行的块
                if content.strip():
                    block = DocumentBlock(
                        type=block_type,
                        content=content,
                        line_start=start_line,
                        line_end=start_line + (end_i - i),
                        page_num=page_num
                    )
                    
                    doc.add_block(block)
                
                line_number += (end_i - i + 1)
                i = end_i + 1
        
        return doc
    
    def _detect_block_type(self, lines: List[str], start: int) -> Tuple[BlockType, int]:
        """检测块类型和结束位置"""
        if start >= len(lines):
            return BlockType.TEXT, start
        
        line = lines[start].strip()
        
        # 跳过空行 - 不创建空的文本块
        if not line:
            # 寻找下一个非空行
            next_non_empty = start + 1
            while next_non_empty < len(lines) and not lines[next_non_empty].strip():
                next_non_empty += 1
            return BlockType.TEXT, next_non_empty - 1  # 返回空行序列的末尾
        
        # 标题
        if self.patterns['heading'].match(line):
            return BlockType.HEADING, start
        
        # 代码围栏
        if self.patterns['code_fence'].match(line):
            end = self._find_code_fence_end(lines, start)
            return BlockType.CODE_FENCE, end
        
        # 图片
        if self.patterns['image'].match(line):
            return BlockType.IMAGE, start
        
        # 图表标题
        if self.patterns['figure_caption'].match(line):
            return BlockType.FIGURE_CAPTION, start
        
        # 表格
        if self.patterns['table_markdown'].match(line):
            end = self._find_table_end(lines, start)
            return BlockType.TABLE, end
        
        # 引用
        if self.patterns['reference'].match(line):
            return BlockType.REFERENCE, start
        
        # 默认是文本
        return BlockType.TEXT, start
    
    def _find_code_fence_end(self, lines: List[str], start: int) -> int:
        """找到代码围栏的结束位置"""
        fence_match = self.patterns['code_fence'].match(lines[start])
        if not fence_match:
            return start
        
        fence_chars = fence_match.group(1)
        
        for i in range(start + 1, len(lines)):
            if re.match(rf'^\s*{re.escape(fence_chars)}', lines[i]):
                return i
        
        return len(lines) - 1  # 未闭合，到文件末尾
    
    def _find_table_end(self, lines: List[str], start: int) -> int:
        """找到表格的结束位置"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            # 空行结束表格
            if not line:
                return i - 1
            # 表格相关的行：包含 | 或者是分隔符
            if '|' in line or self._is_table_separator(line):
                i += 1
                continue
            # 其他情况结束表格
            return i - 1
        return len(lines) - 1
    
    def _is_table_separator(self, line: str) -> bool:
        """检测是否是表格分隔符行"""
        line = line.strip()
        # 标准 markdown 表格分隔符：|---|---|---|
        return bool(re.match(r'^\|[\s\-\|]+\|$', line)) and '-' in line


# ============================================================================
# 处理器：特定功能的简单处理
# ============================================================================

class ReferenceProcessor:
    """引用处理器 - 统一处理各种引用格式"""
    
    def __init__(self):
        self.patterns = {
            'reference_def': re.compile(r"^\[(\d+)\]((?:(?!\[\d+\])[^\n])*)\n(?=^\[\d+\]|$)", re.MULTILINE),
            'reference_range': re.compile(r"\[(\d+)\-(\d+)\]"),
            'reference_multi': re.compile(r"\[(\d+(?:,\s*\d+)*)\]"),  # [1, 2, 3] 格式
            'reference_single': re.compile(r"\[(\d+)\]")
        }
    
    def fix_references(self, text: str) -> str:
        """修复引用格式"""
        # 引用定义
        all_references = re.findall(self.patterns['reference_def'], text)
        for match in all_references:
            original = f"[{match[0]}] {match[1].strip()}"
            replacement = f"[^{match[0]}]: {match[1].strip()}\n"
            text = text.replace(original, replacement)
        
        # 引用范围 [1-5]
        all_ranges = re.findall(self.patterns['reference_range'], text)
        for match in all_ranges:
            original = f"[{match[0]}-{match[1]}]"
            range_refs = " ".join([f"[^{i}]" for i in range(int(match[0]), int(match[1]) + 1)])
            text = text.replace(original, range_refs)
        
        # 多引用 [1, 2, 3] 格式
        all_multis = re.findall(self.patterns['reference_multi'], text)
        for match in all_multis:
            original = f"[{match}]"
            # 分割并转换
            numbers = [n.strip() for n in match.split(',')]
            multi_refs = " ".join([f"[^{n}]" for n in numbers])
            text = text.replace(original, multi_refs)
        
        # 单个引用（最后处理，避免被前面处理了的重复处理）
        all_singles = re.findall(self.patterns['reference_single'], text)
        for match in all_singles:
            original = f"[{match}]"
            replacement = f"[^{match}]"
            # 检查是否已经被处理过
            if original in text and not f"[^{match}]" in text.replace(replacement, ""):
                text = text.replace(original, replacement)
        
        return text


class TitleProcessor:
    """标题处理器 - 统一处理各种标题格式"""
    
    def __init__(self):
        self.patterns = {
            'roman_with_sec': re.compile(r'^(#{1,6})?\s*(Sec(?:tion)?\.\s+)?([IVX]+)\.\s+(.+)$'),
            'number': re.compile(r'^(#{1,6})?\s*(\d+(?:\.\d+)*?)\.\s+(.+)$'),
            'letter_upper': re.compile(r'^(#{1,6})?\s*([A-Z])\.\s+(.+)$'),
            'letter_lower': re.compile(r'^(#{1,6})?\s*([a-z])\.\s+(.+)$')
        }
    
    def fix_titles(self, text: str) -> str:
        """修复标题格式"""
        lines = text.split('\n')
        new_lines = []
        
        # 检测是否有罗马数字
        has_roman = bool(re.search(r'^#{1,6}?\s*(?:Sec(?:tion)?\.\s+)?[IVX]+\.\s+', text, re.MULTILINE))
        
        for line in lines:
            modified = False
            
            # 罗马数字标题
            if has_roman:
                match = self.patterns['roman_with_sec'].match(line)
                if match:
                    section_prefix = match.group(2) or ''
                    roman_num = match.group(3)
                    title = match.group(4)
                    new_line = f"# {section_prefix}{roman_num}. {title}"
                    new_lines.append(new_line)
                    modified = True
            
            # 数字编号
            if not modified:
                match = self.patterns['number'].match(line)
                if match:
                    number = match.group(2)
                    title = match.group(3)
                    level = len(number.split('.'))
                    if has_roman:
                        level += 1
                    new_hashes = '#' * min(level, 6)
                    new_line = f"{new_hashes} {number}. {title}"
                    new_lines.append(new_line)
                    modified = True
            
            # 字母编号
            if not modified:
                for pattern_name in ['letter_upper', 'letter_lower']:
                    match = self.patterns[pattern_name].match(line)
                    if match and not re.match(r'^[A-Z][a-z]', match.group(3)):
                        letter = match.group(2)
                        title = match.group(3)
                        level = 3 if pattern_name == 'letter_upper' else 4
                        new_hashes = '#' * level
                        new_line = f"{new_hashes} {letter}. {title}"
                        new_lines.append(new_line)
                        modified = True
                        break
            
            if not modified:
                new_lines.append(line)
        
        return '\n'.join(new_lines)


class PseudocodeProcessor:
    """伪代码处理器"""
    
    def __init__(self):
        self.alg_header_pattern = re.compile(
            r'^\s*(\*\*)?\s*(Algorithm|算法)\s+([A-Za-z0-9.-]+)?\s*(\*\*)?\s*(.*)$',
            re.IGNORECASE
        )
    
    def wrap_pseudocode_blocks(self, text: str, lang: str = "pseudo") -> str:
        """包装伪代码块"""
        lines = text.splitlines()
        out = []
        i = 0
        in_fence = False
        
        while i < len(lines):
            line = lines[i]
            
            # 维护已有代码块围栏
            if line.strip().startswith("```"):
                in_fence = not in_fence
                out.append(line)
                i += 1
                continue
            
            if (not in_fence) and self.alg_header_pattern.match(line):
                # 捕获算法块
                header_line = line
                block = [header_line]
                i += 1
                
                # 向下收集
                while i < len(lines):
                    peek = lines[i]
                    if peek.strip().startswith("```") or re.match(r'^\s*#{1,6}\s', peek):
                        break
                    if not self._is_algo_continuation(peek):
                        break
                    block.append(peek)
                    i += 1
                
                # 输出fenced block
                out.append(f"```{lang}")
                
                # 处理标题行
                m = self.alg_header_pattern.match(header_line)
                if m:
                    alg_no = (m.group(3) or "").strip()
                    rest = (m.group(5) or "").strip()
                    title = f"Algorithm {alg_no}: {rest}".strip() if alg_no or rest else "Algorithm"
                    out.append(f"// {self._clean_inline(title)}")
                
                for raw in block[1:]:
                    s = raw.strip()
                    if s == "***":
                        out.append("// " + "-" * 40)
                        continue
                    out.append(self._clean_inline(raw))
                
                out.append("```")
                continue
            
            out.append(line)
            i += 1
        
        return "\n".join(out)
    
    def _is_algo_continuation(self, line: str) -> bool:
        """判断是否是算法块的继续"""
        s = line.strip()
        if s == "" or s == "***":
            return True
        if re.match(r'^\s*\d+\s*[:.)]\ ', s):
            return True
        if re.match(r'^\s*(function|procedure|for|while|if|else|repeat|return|end)\b', s, re.I):
            return True
        return False
    
    def _clean_inline(self, text: str) -> str:
        """清理内联格式"""
        text = re.sub(r'<\s*sub\s*>\s*(.*?)\s*<\s*/\s*sub\s*>', lambda m: '_' + re.sub(r'\*', '', m.group(1)), text, flags=re.I)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        return text


# ============================================================================
# 主处理器：全新的简化版本
# ============================================================================

class RepairFSM:
    """
    简洁的修复状态机 - 专门处理复杂分割场景
    
    5个核心状态，清晰的转换，专注实际问题
    """
    
    def __init__(self):
        self.state = 'SCANNING'
        self.current_idx = 0
        self.repair_count = 0
    
    def repair_document(self, doc: Document) -> Document:
        """状态机主循环 - 简洁明了"""
        blocks = doc.blocks.copy()
        self.current_idx = 0
        
        while self.current_idx < len(blocks) - 2:
            
            if self.state == 'SCANNING':
                # 扫描寻找分割模式
                if self._found_split_pattern(blocks):
                    self.state = 'ANALYZING'
                else:
                    self.current_idx += 1
            
            elif self.state == 'ANALYZING':
                # 分析分割类型和修复策略
                repair_action = self._analyze_split_type(blocks)
                if repair_action:
                    self.state = 'REPAIRING'  
                    self.repair_action = repair_action
                else:
                    self.state = 'SCANNING'
                    self.current_idx += 1
            
            elif self.state == 'REPAIRING':
                # 执行修复
                blocks = self._execute_repair(blocks, self.repair_action)
                self.repair_count += 1
                self.state = 'SCANNING'
                # current_idx 由修复函数调整
            
        logging.info(f"FSM修复了 {self.repair_count} 个复杂分割")
        return Document(blocks=blocks, metadata=doc.metadata)
    
    def _found_split_pattern(self, blocks: List[DocumentBlock]) -> bool:
        """检测是否找到分割模式"""
        if self.current_idx + 2 >= len(blocks):
            return False
            
        before = blocks[self.current_idx]
        middle = blocks[self.current_idx + 1]
        after = blocks[self.current_idx + 2]
        
        # 经典分割模式：文本 + 非文本 + 文本
        return (before.type == BlockType.TEXT and 
                middle.type != BlockType.TEXT and
                after.type == BlockType.TEXT)
    
    def _analyze_split_type(self, blocks: List[DocumentBlock]) -> Optional[Dict]:
        """分析分割类型 - 5种核心场景"""
        before = blocks[self.current_idx]
        middle = blocks[self.current_idx + 1] 
        after = blocks[self.current_idx + 2]
        
        before_text = before.content.strip()
        after_text = after.content.strip()
        
        # 场景1: 图片分割段落
        if middle.type in [BlockType.IMAGE, BlockType.FIGURE_CAPTION]:
            if self._is_paragraph_split(before_text, after_text):
                return {
                    'type': 'image_split',
                    'confidence': self._calculate_split_confidence(before_text, after_text),
                    'strategy': 'merge_around_image'
                }
        
        # 场景2: 表格分割段落  
        elif middle.type == BlockType.TABLE:
            if self._is_paragraph_split(before_text, after_text):
                return {
                    'type': 'table_split', 
                    'confidence': self._calculate_split_confidence(before_text, after_text),
                    'strategy': 'merge_around_table'
                }
        
        # 场景3: 跨页分割
        elif middle.type == BlockType.PAGE_BOUNDARY:
            if before_text.endswith('-') and after_text and after_text[0].islower():
                return {
                    'type': 'hyphen_split',
                    'confidence': 0.95,
                    'strategy': 'merge_hyphenated'
                }
            elif self._is_sentence_continuation(before_text, after_text):
                return {
                    'type': 'sentence_split',
                    'confidence': 0.8,
                    'strategy': 'merge_sentence'
                }
        
        # 场景4: 代码块分割 
        elif middle.type == BlockType.CODE_FENCE:
            if self._is_paragraph_split(before_text, after_text):
                return {
                    'type': 'code_split',
                    'confidence': 0.6,
                    'strategy': 'merge_around_code'
                }
        
        # 场景5: 其他块类型分割
        else:
            if self._is_strong_paragraph_split(before_text, after_text):
                return {
                    'type': 'other_split',
                    'confidence': 0.5, 
                    'strategy': 'merge_conservative'
                }
        
        return None
    
    def _execute_repair(self, blocks: List[DocumentBlock], action: Dict) -> List[DocumentBlock]:
        """执行修复动作"""
        before_idx = self.current_idx
        middle_idx = self.current_idx + 1
        after_idx = self.current_idx + 2
        
        before = blocks[before_idx]
        middle = blocks[middle_idx]
        after = blocks[after_idx]
        
        if action['confidence'] < 0.7:
            # 低置信度，跳过
            self.current_idx += 1
            return blocks
        
        # 根据策略执行修复
        if action['strategy'] in ['merge_around_image', 'merge_around_table', 'merge_around_code']:
            # 合并前后文本，保留中间元素
            merged_text = before.content.rstrip() + ' ' + after.content.lstrip()
            merged_block = DocumentBlock(
                type=BlockType.TEXT,
                content=merged_text,
                line_start=before.line_start,
                line_end=after.line_end,
                source="fsm_repair"
            )
            
            # 重建blocks: merged_block + middle + 其余部分
            new_blocks = (blocks[:before_idx] + 
                         [merged_block, middle] +  
                         blocks[after_idx + 1:])
            
            self.current_idx = before_idx + 1  # 跳过已处理的部分
            
        elif action['strategy'] == 'merge_hyphenated':
            # 连字符合并，移除页面边界
            merged_text = before.content.rstrip('-') + after.content.lstrip()
            merged_block = DocumentBlock(
                type=BlockType.TEXT,
                content=merged_text,
                line_start=before.line_start,
                line_end=after.line_end,
                source="fsm_hyphen_repair"
            )
            
            new_blocks = (blocks[:before_idx] +
                         [merged_block] +
                         blocks[after_idx + 1:])
            
            self.current_idx = before_idx  # 继续从这里扫描
            
        elif action['strategy'] == 'merge_sentence':
            # 句子合并，移除页面边界
            merged_text = before.content.rstrip() + ' ' + after.content.lstrip()
            merged_block = DocumentBlock(
                type=BlockType.TEXT,
                content=merged_text,
                line_start=before.line_start,
                line_end=after.line_end,
                source="fsm_sentence_repair"
            )
            
            new_blocks = (blocks[:before_idx] +
                         [merged_block] +
                         blocks[after_idx + 1:])
            
            self.current_idx = before_idx
            
        else:
            # 保守合并
            self.current_idx += 1
            return blocks
        
        return new_blocks
    
    def _is_paragraph_split(self, before_text: str, after_text: str) -> bool:
        """判断是否是段落分割"""
        # 前文未结束 + 后文小写开头 + 语义相关
        incomplete = not re.search(r'[.!?。！？]\s*$', before_text)
        lowercase_start = after_text and after_text[0].islower()
        semantic_related = len(set(before_text.lower().split()) & set(after_text.lower().split())) > 1
        
        return incomplete and (lowercase_start or semantic_related)
    
    def _is_sentence_continuation(self, before_text: str, after_text: str) -> bool:
        """判断是否是句子延续"""
        return (not re.search(r'[.!?。！？]\s*$', before_text) and
                after_text and after_text[0].islower())
    
    def _is_strong_paragraph_split(self, before_text: str, after_text: str) -> bool:
        """强段落分割信号"""
        return (self._is_paragraph_split(before_text, after_text) and
                len(set(before_text.lower().split()) & set(after_text.lower().split())) > 2)
    
    def _calculate_split_confidence(self, before_text: str, after_text: str) -> float:
        """计算分割修复的置信度"""
        confidence = 0.0
        
        # 前文未完成
        if not re.search(r'[.!?。！？]\s*$', before_text):
            confidence += 0.3
        
        # 后文小写开头
        if after_text and after_text[0].islower():
            confidence += 0.4
        
        # 词汇重叠
        common_words = len(set(before_text.lower().split()) & set(after_text.lower().split()))
        if common_words > 1:
            confidence += min(0.3, common_words * 0.1)
        
        return min(confidence, 1.0)


class MarkdownFixer:
    """
    Markdown修复器 - Linus风格重构版
    
    原则：
    1. 使用正确的数据结构
    2. 消除所有特殊情况  
    3. 保持简单明了
    4. 用简洁的FSM处理复杂场景
    """
    
    def __init__(self):
        self.parser = DocumentParser()
        self.ref_processor = ReferenceProcessor()
        self.title_processor = TitleProcessor()
        self.pseudo_processor = PseudocodeProcessor()
        self.repair_fsm = RepairFSM()  # 添加FSM处理复杂场景
    
    def fix(self, text: str) -> str:
        """主修复方法 - 四步流水线"""
        # Step 1: 解析
        doc = self.parser.parse(text)
        
        # Step 2: 简单合并
        merged_count = doc.merge_consecutive_blocks()
        
        # Step 3: FSM复杂修复 ⭐新增⭐
        doc = self.repair_fsm.repair_document(doc)
        
        # Step 4: 输出
        result = doc.to_markdown()
        
        # 后处理
        result = self.title_processor.fix_titles(result)
        result = self.ref_processor.fix_references(result)
        result = self.pseudo_processor.wrap_pseudocode_blocks(result)
        result = self._clean_whitespace(result)
        
        logging.info(f"基础合并: {merged_count} 个块，FSM修复: {self.repair_fsm.repair_count} 个复杂分割")
        
        return result
    
    def _clean_whitespace(self, text: str) -> str:
        """清理多余空格"""
        # 替换HTML实体
        text = text.replace("&nbsp;", " ")
        text = text.replace("&quot;", '"')
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        
        # 清理多余空行
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text


# ============================================================================
# 兼容性接口
# ============================================================================

def fix_content(markdown_content: str) -> str:
    """主接口 - 保持向后兼容"""
    fixer = MarkdownFixer()
    return fixer.fix(markdown_content)


# ============================================================================
# 命令行工具函数
# ============================================================================

def fix_file(markdown_file: str, output_file: str, level: str = "moderate"):
    """修复文件的命令行接口"""
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
        type=str,
        default="moderate",
        help="Fix level: conservative/moderate/aggressive",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_file = args.output_file
    if output_file is None:
        output_file = args.markdown_file.replace(".md", "_fixed.md")

    fix_file(args.markdown_file, output_file, args.level)