#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panflute filter: Move figure/table clusters to the head of their nearest section.

Behavior
- A "cluster" = one or more figure-like blocks (image-only paragraph, Markdown table,
  or HTML <table> raw block) optionally followed by a caption paragraph
  (prefix matches: Fig./Figure/Figs./Table/Tab./图/表 + number).
- Operates per section (Header level 1..6). Root (no header) is ignored.
- Clusters already at the start of a section (immediately after header) are kept in place.
- Nested structures (lists, block quotes, code fences) are not modified.

Usage:
  pandoc in.md -f gfm -t gfm -F "python3 move_figures_to_header.py" -o out.md
"""

import re
import panflute as pf
from typing import List, Union

CAPTION_RE = re.compile(
    r'^(?:fig\.|figure|figs\.|table|tab\.|图|表)\s*[\d]+', re.IGNORECASE
)

# ---------- Predicates on Blocks ----------

def is_image_only_para(b: pf.Element) -> bool:
    if not isinstance(b, pf.Para):
        return False
    has_img = False
    for x in b.content:
        if isinstance(x, pf.Image):
            has_img = True
        elif isinstance(x, (pf.Space, pf.LineBreak, pf.SoftBreak)):
            continue
        else:
            # Any other inline (Str, Emph, etc.) makes it not "image-only"
            return False
    return has_img

def is_markdown_table(b: pf.Element) -> bool:
    return isinstance(b, pf.Table)

def is_html_table_raw(b: pf.Element) -> bool:
    return isinstance(b, pf.RawBlock) and b.format == 'html' and '<table' in b.text.lower()

def is_figure_like(b: pf.Element) -> bool:
    return is_image_only_para(b) or is_markdown_table(b) or is_html_table_raw(b)

def is_caption_para(b: pf.Element) -> bool:
    if not isinstance(b, pf.Para):
        return False
    txt = pf.stringify(b).strip()
    return bool(CAPTION_RE.match(txt))

# ---------- Section Tree ----------

class Section:
    def __init__(self, level: int, header: Union[pf.Header, None]):
        self.level = level
        self.header = header              # None for root
        self.children: List[Union['Section', pf.Element]] = []

def build_section_tree(doc: pf.Doc) -> Section:
    root = Section(0, None)
    stack: List[Section] = [root]

    for blk in list(doc.content):
        if isinstance(blk, pf.Header):
            level = blk.level
            while stack and stack[-1].level >= level:
                stack.pop()
            sec = Section(level, blk)
            stack[-1].children.append(sec)
            stack.append(sec)
        else:
            stack[-1].children.append(blk)
    return root

# ---------- Cluster detection & reordering (per section) ----------

def find_clusters(children: List[Union[Section, pf.Element]]):
    """
    Return list of (start_idx, end_idx_exclusive) for clusters among top-level Blocks.
    Clusters are contiguous figure-like blocks optionally followed by a single caption paragraph.
    Also supports "caption-first" pattern (caption + figure-like...).
    """
    clusters = []
    i = 0
    n = len(children)
    while i < n:
        item = children[i]
        # Skip nested sections
        if isinstance(item, Section):
            i += 1
            continue

        # caption-first cluster
        if is_caption_para(item):
            j = i + 1
            took_fig = False
            while j < n and not isinstance(children[j], Section) and is_figure_like(children[j]):
                j += 1
                took_fig = True
            if took_fig:
                clusters.append((i, j))
                i = j
                continue

        # figure-first cluster
        if is_figure_like(item):
            j = i + 1
            while j < n and not isinstance(children[j], Section) and is_figure_like(children[j]):
                j += 1
            # optional trailing caption
            if j < n and not isinstance(children[j], Section) and is_caption_para(children[j]):
                j += 1
            clusters.append((i, j))
            i = j
            continue

        i += 1
    return clusters

def reorder_section(sec: Section):
    """Reorder clusters to the head of this section; then recurse on nested sections."""
    # Recurse first on nested sections that appear before we reshuffle blocks
    for ch in sec.children:
        if isinstance(ch, Section):
            reorder_section(ch)

    # Root has no header → do not move at root-level
    if sec.header is None:
        return

    children = sec.children

    # Identify cluster ranges
    clusters = find_clusters(children)
    if not clusters:
        return

    # Determine which clusters are already at the very start (index 0)
    # We treat "start" strictly: must be the very first items after header, i.e., children index 0.
    front_clusters = []
    consumed = 0
    for (s, e) in clusters:
        if s == consumed:
            front_clusters.append((s, e))
            consumed = e
        else:
            break

    # Blocks to extract (front + moved) as lists for stable re-insertion
    def collect_blocks(ranges):
        picked = []
        for (s, e) in ranges:
            for k in range(s, e):
                picked.append(children[k])
        return picked

    # Clusters that should move = clusters excluding front_clusters
    movable_clusters = clusters[len(front_clusters):]

    if not movable_clusters:
        # Nothing to move: already at head
        return

    front_blocks = collect_blocks(front_clusters)
    moved_blocks = collect_blocks(movable_clusters)

    # Build "remaining children" by removing all blocks that belong to any cluster
    cluster_member_ids = set(id(x) for x in (front_blocks + moved_blocks))
    remaining = [x for x in children if id(x) not in cluster_member_ids]

    # New order: [front clusters (as-is)] + [moved clusters (in original order)] + [the rest]
    sec.children = front_blocks + moved_blocks + remaining

    # Recurse again on nested sections that may have shifted positions (safe: objects preserved)
    for ch in sec.children:
        if isinstance(ch, Section):
            reorder_section(ch)

# ---------- Flatten back to document ----------

def flatten_section(sec: Section, out: List[pf.Element]):
    if sec.header is not None:
        out.append(sec.header)
    for ch in sec.children:
        if isinstance(ch, Section):
            flatten_section(ch, out)
        else:
            out.append(ch)

# ---------- Panflute hooks ----------

def finalize(doc: pf.Doc):
    tree = build_section_tree(doc)
    reorder_section(tree)
    new_blocks: List[pf.Element] = []
    flatten_section(tree, new_blocks)
    doc.content = new_blocks
    
def passthrough(elem, doc):
    # 不做任何元素级修改；仅为了满足 run_filters 的 actions 形参
    return None

if __name__ == "__main__":
    pf.run_filters([passthrough], finalize=finalize)
