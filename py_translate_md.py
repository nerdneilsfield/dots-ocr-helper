#! /usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from string import Template
import sys
import json
import re
import base64
import dataclasses
import logging
from textwrap import dedent
import time
import ollama
import toml
import yaml
from pathlib import Path
from tokenize import PlainToken
from typing import Dict, List, Any, Tuple, Optional

try:
    import coloredlogs

    coloredlogs.install(level=logging.INFO)
except ImportError:
    pass


@dataclasses.dataclass
class TranslateConfig:
    source_lang: Optional[str] = None  # e.g., "en"
    target_lang: str = "zh"  # e.g., "zh", "en"
    max_chunk_chars: int = 4000
    translate_tables: bool = False  # False=整表冻结；True=仅冻结表结构，开放单元格文本
    translate_links_text: bool = False  # False=链接整体冻结；True=仅翻译锚文本，冻结URL
    translate_image_alt: bool = False  # 是否翻译图片 alt 文本
    keep_node_markers: bool = True  # 安全起见，保持为 True
    strict_placeholder_check: bool = True  # 严格校验占位符一致性
    retry_failed_nodes: bool = True  # 对失败节点单独重试
    retry_times: int = 3
    parallel_groups: int = 1
    log_level: str = "INFO"
    translator_type: str = ""


def create_default_cfg() -> TranslateConfig:
    return TranslateConfig(
        source_lang="en",  # e.g., "en"
        target_lang="zh",  # e.g., "zh", "en"
        max_chunk_chars=3000,
        translate_tables=False,  # False=整表冻结；True=仅冻结表结构，开放单元格文本
        translate_links_text=False,  # False=链接整体冻结；True=仅翻译锚文本，冻结URL
        translate_image_alt=False,  # 是否翻译图片 alt 文本
        keep_node_markers=True,  # 安全起见，保持为 True
        strict_placeholder_check=True,  # 严格校验占位符一致性
        retry_failed_nodes=True,  # 对失败节点单独重试
        retry_times=3,
        parallel_groups=1,
        log_level="INFO",
        translator_type="echo",
    )


def load_configs(file_path: str) -> tuple[TranslateConfig, dict]:
    extension = os.path.splitext(file_path)[1]
    default_cfg = create_default_cfg()
    if extension == ".json":
        with open(file_path, "r") as f:
            loaded_cfg = json.load(f)
            assert isinstance(loaded_cfg, dict), "loaded_cfg is not a dict"
            for key in loaded_cfg.keys():
                if key in default_cfg.__dataclass_fields__.keys():
                    setattr(default_cfg, key, loaded_cfg[key])
            translator_cfg = loaded_cfg.get("translator_config", {})
            return default_cfg, translator_cfg
    elif extension == ".yaml" or extension == ".yml":
        with open(file_path, "r") as f:
            loaded_cfg = yaml.load(f)
            assert isinstance(loaded_cfg, dict), "loaded_cfg is not a dict"
            for key in loaded_cfg.keys():
                if key in default_cfg.__dataclass_fields__.keys():
                    setattr(default_cfg, key, loaded_cfg[key])
            translator_cfg = loaded_cfg.get("translator_config", {})
            return default_cfg, translator_cfg
    elif extension == ".toml":
        with open(file_path, "r") as f:
            loaded_cfg = toml.load(f)
            assert isinstance(loaded_cfg, dict), "loaded_cfg is not a dict"
            for key in loaded_cfg.keys():
                if key in default_cfg.__dataclass_fields__.keys():
                    setattr(default_cfg, key, loaded_cfg[key])
            translator_cfg = loaded_cfg.get("translator_config", {})
            return default_cfg, translator_cfg
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


class PlaceHolderStore:
    """
    Store placeholders for text.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._rev: dict[str, str] = {}
        self._kind_count: dict[str, int] = {}

        self.length = 0

    def add(self, kind: str, text: str) -> str:
        if text in self._map:
            return self._map[text]

        self.length += 1
        length_str = str(self.length).zfill(6)
        ph = f"__PH_{kind}_{length_str}__"
        self._map[text] = ph
        self._rev[ph] = text
        self._kind_count[kind] = self._kind_count.get(kind, 0) + 1

        return self._map[text]

    def save(self, file_path: str) -> None:
        save_list = {}
        save_list["map"] = self._map
        save_list["rev"] = self._rev
        save_list["kind_count"] = self._kind_count

        with open(file_path, "w") as f:
            json.dump(save_list, f, indent=4)

    def load(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            save_list = json.load(f)

        self._map = save_list["map"]
        self._rev = save_list["rev"]
        self._kind_count = save_list["kind_count"]

        self.length = len(self._map)

    def restore_all(self, text: str) -> str:
        for ph, raw in sorted(self._rev.items(), key=lambda x: -len(x[0])):
            text = text.replace(ph, raw)
        return text

    def contains_all(self, text: str) -> bool:
        return all(ph in text for ph in self._map.keys())

    def diff_missing(self, text: str) -> List[str]:
        return [ph for ph in self._map.keys() if ph not in text]

    def snapshot(self) -> Dict[str, str]:
        return dict(self._map)


def _cdata_wrap(s: str) -> str:
    # 安全处理 CDATA 终止符：把每个 "]]>" 切开，避免破坏 XML
    return s.replace("]]>", "]]]]><![CDATA[>")


SYSTEM_RULES = dedent(
    """\
You are a translation engine. Follow these invariant rules:
- Preserve all original formatting exactly (Markdown, whitespace, paragraph breaks).
- Do NOT translate LaTeX ($...$, $$...$$, \\( ... \\), \\[ ... \\]) or LaTeX commands/environments.
- Keep all HTML tags intact.
- Do NOT alter abbreviations, technical terms, or code identifiers.
- Handle two NODE styles: @@NODE_START_{n}@@/@@NODE_END_{n}@@ and <NODE_START_{n}></NODE_END_{n}>.
- Respect PRESERVE spans: @@PRESERVE_{n}@@ ... @@/PRESERVE_{n}@@ (leave markers and enclosed text unchanged).
- Placeholders like __PH_[A-Z0-9_]+__ must remain unchanged.
- Output ONLY the NODE blocks in original order; no extra commentary.
- If markers malformed: reproduce original block verbatim and append <!-- VIOLATION: reason --> once.
"""
)

FIRST_TRANSLATE_XML_TPL = Template(
    dedent(
        """\
<TranslationTask version="1.0">
  <meta>
    <source_lang>$SOURCE_LANG</source_lang>
    <target_lang>$TARGET_LANG</target_lang>
    <visibility_note>Sections with visibility="internal" are instructions and MUST NOT appear in the final output.</visibility_note>
  </meta>

  <constraints visibility="internal">
    <rule id="fmt-1">Preserve ALL original formatting exactly: Markdown, whitespace, line breaks, paragraph spacing.</rule>
    <rule id="fmt-2">Do NOT translate any content inside LaTeX ($$...$$, $$$$...$$$$, \\( ... \\), \\[ ... \\]) or LaTeX commands/environments.</rule>
    <rule id="fmt-3">Keep ALL HTML tags intact.</rule>
    <rule id="fmt-4">Do NOT alter abbreviations, technical terms, or code identifiers; translate surrounding prose only.</rule>
    <rule id="fmt-5">Document structure must be preserved, including blank lines (double newlines) between blocks.</rule>
  </constraints>

  <markers visibility="internal">
    <preserve>
      <open>@@PRESERVE_{n}@@</open>
      <close>@@/PRESERVE_{n}@@</close>
      <instruction>Leave both markers and enclosed text EXACTLY unchanged.</instruction>
    </preserve>
    <node accepted_styles="double">
      <style type="at">
        <open>@@NODE_START_{n}@@</open>
        <close>@@NODE_END_{n}@@</close>
      </style>
      <style type="angle">
        <open>&lt;NODE_START_{n}&gt;</open>
        <close>&lt;/NODE_END_{n}&gt;</close>
      </style>
      <scope>Translate ONLY the text inside each NODE block.</scope>
      <layout>
        <rule>Preserve the exact presence/absence of newlines around the content: if input has newlines after START and before END, keep them; if single-line, keep single-line.</rule>
        <rule>Preserve all spaces and blank lines BETWEEN NODE blocks exactly.</rule>
      </layout>
    </node>
    <placeholders>
      <pattern>__PH_[A-Z0-9_]+__</pattern>
      <instruction>All placeholders matching this regex MUST be left unchanged (e.g., __PH_FOOTREF_000195__).</instruction>
    </placeholders>
  </markers>

  <output_spec visibility="internal">
    <rule id="out-1">Output ONLY the NODE blocks in the original order. Non-NODE text must NOT be echoed unless it is part of a NODE.</rule>
    <rule id="out-2">For each NODE: emit the exact START marker, then the translated content, then the exact END marker, preserving surrounding whitespace/line breaks as in input.</rule>
    <rule id="out-3">Do NOT reveal or restate any instructions with visibility="internal".</rule>
  </output_spec>

  <quality_checks visibility="internal">
    <check>Count of START and END NODE markers is identical to input; indices {n} match 1:1.</check>
    <check>No PRESERVE spans were altered; byte-for-byte identical.</check>
    <check>No LaTeX/HTML/code tokens changed; only prose translated.</check>
    <check>Paragraph breaks (double newlines) and intra-block whitespace unchanged.</check>
  </quality_checks>

  <fallback visibility="internal">
    <strategy>If a block violates constraints or markers are malformed, do NOT guess. Reproduce the original block unchanged and append a single-line comment <!-- VIOLATION: reason --> after the block.</strategy>
  </fallback>

  <io>
    <input>
      <![CDATA[
$TEXT_TO_TRANSLATE
      ]]>
    </input>
    <expected_output visibility="internal">
      <note>Emit only transformed NODE blocks per output_spec. Nothing else.</note>
    </expected_output>
  </io>
</TranslationTask>
"""
    )
)


class BaseTranslator:

    def __init__(self, name: str) -> None:
        self.translator_name = name

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    def translate_with_retry(
        self, text: str, source_lang: str, target_lang: str, timeout: int = 7200
    ) -> str:
        raise NotImplementedError


class EchoTranslator(BaseTranslator):
    def __init__(self, configs: dict) -> None:
        super().__init__("echo")
        self.configs = configs

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return text

    def translate_with_retry(
        self, text: str, source_lang: str, target_lang: str, timeout: int = 7200
    ) -> str:
        return text


class OllamaTranslator(BaseTranslator):
    def __init__(self, configs: dict) -> None:
        super().__init__("ollama")
        try:
            self.host = configs.get("host", "http://localhost:11434")
            self.model = configs["model"]
            self.retry_times = configs.get("retry_times", 3)
            self.timeout = configs.get("timeout", 7200)
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError("ollama host or model not found")

        self.client = ollama.Client(host=self.host)

        print(f"connected to ollama with host {self.host} and model {self.model}")

    def translate(
        self, text: str, source_lang: str, target_lang: str, timeout: int = None
    ) -> str:

        user_xml = FIRST_TRANSLATE_XML_TPL.substitute(
            SOURCE_LANG=source_lang,
            TARGET_LANG=target_lang,
            TEXT_TO_TRANSLATE=_cdata_wrap(text),
        )
        messages = [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": user_xml},
        ]

        response = self.client.chat(model=self.model, messages=messages)

        # print(response)

        return response.message.content

    def translate_with_retry(
        self, text: str, source_lang: str, target_lang: str, timeout: int = None
    ) -> str:
        retry_times = 0
        if timeout is None:
            timeout = self.timeout
        while retry_times < self.retry_times:
            try:
                return self.translate(text, source_lang, target_lang)
            except Exception as e:
                print(
                    f"ollama translate failed {retry_times + 1} / {self.retry_times} , reason: {e}"
                )
                retry_times += 1
                time.sleep(1 + 1.5 * retry_times)
        return text


@dataclasses.dataclass
class Node:
    nid: int
    origin_text: str
    translated_text: str


def remove_unnecessary_whitespaces(text: str) -> str:
    """
    移除不必要的空格
    """
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


class MarkdownProtector:

    # 常见“块级”HTML 标签（覆盖 CommonMark/GFM 常见集合）
    BLOCK_HTML_TAGS = (
        "address|article|aside|blockquote|body|caption|center|col|colgroup|dd|details|dialog|div|dl|dt|fieldset|"
        "figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|legend|li|link|main|menu|nav|"
        "noframes|ol|optgroup|option|p|param|section|summary|table|tbody|td|tfoot|th|thead|title|tr|ul|video|audio|canvas"
    )
    # HTML5 空元素（自闭合）
    VOID_HTML_TAGS = (
        "area|base|br|col|embed|hr|img|input|keygen|link|meta|param|source|track|wbr"
    )

    def __init__(self) -> None:
        pass

    def protect(self, text: str, cfg: TranslateConfig, store: PlaceHolderStore) -> str:
        stage1 = self._partition_by_blocks(text, cfg, store)
        stage2 = self._freeze_inline(stage1, cfg, store)
        return stage2

    def unprotect(self, text: str, place_holder_store: PlaceHolderStore) -> str:
        text = place_holder_store.restore_all(text)
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
        m = re.match(
            rf"^<(?P<tag>{MarkdownProtector.BLOCK_HTML_TAGS})\b", s, flags=re.IGNORECASE
        )
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
        if re.match(
            rf"^(?:{MarkdownProtector.VOID_HTML_TAGS})$", tag, flags=re.IGNORECASE
        ):
            return i

        # 普通块级标签：做简单嵌套计数
        open_pat = re.compile(
            rf"<{tag}\b(?![^>]*?/>)", re.IGNORECASE
        )  # 排除 <tag .../>
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
        s: str, cfg: TranslateConfig, store: PlaceHolderStore
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
            fence = MarkdownProtector._line_starts_with_fence(line)
            if fence:
                j = i + 1
                while j < n and not re.match(rf"^\s*{re.escape(fence)}", lines[j]):
                    j += 1
                # 包含起止两行
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(store.add("CODEFENCE", block))
                    i = j + 1
                    continue
                else:
                    # 未闭合：按剩余全当代码块处理
                    block = "".join(lines[i:])
                    out.append(store.add("CODEFENCE", block))
                    break

            # 2) HTML 代码型块
            tag = MarkdownProtector._line_starts_html_codey(line)
            if tag:
                j = i
                while j < n and not MarkdownProtector._line_ends_html(tag, lines[j]):
                    j += 1
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(store.add("HTMLBLOCK", block))
                    i = j + 1
                    continue
                else:
                    block = "".join(lines[i:])
                    out.append(store.add("HTMLBLOCK", block))
                    break

            # 2b) 通用块级 HTML（整块冻结）
            tag_block = MarkdownProtector._line_starts_block_html_open(line)
            if tag_block:
                j = MarkdownProtector._scan_until_html_block_end(lines, i, tag_block)
                block = "".join(lines[i : j + 1])
                out.append(store.add("HTMLBLOCK", block))
                i = j + 1
                continue

            # 3) 块级数学 $$...$$
            if MarkdownProtector._line_is_block_math_open(line):
                j = i + 1
                while j < n and not MarkdownProtector._line_is_block_math_close(
                    lines[j]
                ):
                    j += 1
                if j < n:
                    block = "".join(lines[i : j + 1])
                    out.append(store.add("MATHBLOCK", block))
                    i = j + 1
                    continue
                else:
                    block = "".join(lines[i:])
                    out.append(store.add("MATHBLOCK", block))
                    break

            # 4) 表格（可配置：整表冻结）
            if not cfg.translate_tables:
                if (
                    i + 1 < n
                    and MarkdownProtector._looks_like_table_header(line)
                    and MarkdownProtector._looks_like_table_delim(lines[i + 1])
                ):
                    j = i + 2
                    # 吸收后续表行，直到遇到空行或非表格行
                    while (
                        j < n
                        and (
                            "|" in lines[j]
                            or MarkdownProtector._looks_like_table_delim(lines[j])
                        )
                        and not MarkdownProtector._is_blank(lines[j])
                    ):
                        j += 1
                    block = "".join(lines[i:j])
                    out.append(store.add("TABLE", block))
                    i = j
                    continue

            # 5) 脚注定义
            if re.match(r"^\[\^[^\]]+\]:", line):
                j = i + 1
                # 吸收后续缩进行
                while j < n and (
                    re.match(r"^\s{4,}", lines[j])
                    or MarkdownProtector._is_blank(lines[j])
                ):
                    j += 1
                block = "".join(lines[i:j])
                out.append(store.add("FOOTDEF", block))
                i = j
                continue

            # 6) 缩进代码块（4 空格起）
            if re.match(r"^( {4}|\t)", line):
                j = i + 1
                while j < n and re.match(r"^( {4}|\t)", lines[j]):
                    j += 1
                block = "".join(lines[i:j])
                out.append(store.add("INDENTCODE", block))
                i = j
                continue

            # 默认：普通行
            out.append(line)
            i += 1

        protected_text = "\n\n".join(out)

        while "\n\n\n" in protected_text:
            protected_text = protected_text.replace("\n\n\n", "\n\n")
        return protected_text

    @staticmethod
    def _freeze_inline(text: str, cfg: TranslateConfig, store: PlaceHolderStore) -> str:
        """
        行内保护：图片、链接、行内代码、行内数学、脚注引用、自动链接/内联HTML等。
        注意：按策略控制是否翻译锚文本、图片 alt。
        """
        s = text

        # 0) 链接定义（形如 [id]: http...），若块级没吸到，这里兜底
        def repl_link_def(m: re.Match) -> str:
            return store.add("LINKDEF", m.group(0))

        s = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", repl_link_def, s, flags=re.MULTILINE)

        # 1) 图片 ![alt](url "title") —— 默认整体冻结；若允许翻 alt，则改为仅冻结 ()，放开 []
        img_pattern = re.compile(r"!\[(?:[^\]\\]|\\.)*?\]\((?:[^()\\]|\\.)*?\)")
        if not cfg.translate_image_alt:
            s = img_pattern.sub(lambda m: store.add("IMAGE", m.group(0)), s)
        else:
            # 冻结括号部分 ()，放开 alt
            def repl_img_alt(m: re.Match) -> str:
                full = m.group(0)
                # 拆分 alt 与 ( )
                m2 = re.match(r"(!\[)(.*?)(\]\()(.+)(\))", full)
                if not m2:
                    return store.add("IMAGE", full)
                head, alt, mid, tail, endp = m2.groups()
                ph = store.add("IMGURL", mid + tail + endp)  # 冻结 () 与内部
                return f"{head}{alt}{ph}"

            s = img_pattern.sub(repl_img_alt, s)

        # 2) 普通链接 [text](url) —— 默认整体冻结；若开启，则仅冻结 (url)，放开 [text]
        link_pattern = re.compile(r"\[(?:[^\]\\]|\\.)*?\]\((?:[^()\\]|\\.)*?\)")
        if not cfg.translate_links_text:
            s = link_pattern.sub(lambda m: store.add("LINK", m.group(0)), s)
        else:

            def repl_link_text(m: re.Match) -> str:
                full = m.group(0)
                m2 = re.match(r"(\[)(.*?)(\]\()(.+)(\))", full)
                if not m2:
                    return store.add("LINK", full)
                lbr, txt, mid, tail, rbr = m2.groups()
                ph = store.add("LINKURL", mid + tail + rbr)
                return f"{lbr}{txt}{ph}"

            s = link_pattern.sub(repl_link_text, s)

        # 3) 引用式链接 [text][id] —— 整体冻结（或按策略仅放开 [text]）
        ref_link_pattern = re.compile(r"\[(?:[^\]\\]|\\.)*?\]\[[^\]]+\]")
        s = ref_link_pattern.sub(lambda m: store.add("REFLINK", m.group(0)), s)

        # 4) 自动链接 <http...> / <mailto:...> —— 冻结
        autolink_pattern = re.compile(r"<(?:https?://|mailto:)[^>]+>")
        s = autolink_pattern.sub(lambda m: store.add("AUTOLINK", m.group(0)), s)

        # 4b) URL
        url_pattern = re.compile(r"(https?://[^ ]+)")
        s = url_pattern.sub(lambda m: store.add("URL", m.group(0)), s)

        # 5) 行内代码 `...`（反引号可变宽度）
        inline_code_pattern = re.compile(r"(?<!`)(`+)([^`\n]+?)\1(?!`)")
        s = inline_code_pattern.sub(lambda m: store.add("CODE", m.group(0)), s)

        # 6) 行内数学 $...$（简化版）
        inline_math_pattern = re.compile(r"\$(?!\s)([^$\n]+?)\$(?!\$)")
        s = inline_math_pattern.sub(lambda m: store.add("MATH", m.group(0)), s)

        # 7) 脚注引用 [^id]
        footref_pattern = re.compile(r"\[\^[^\]]+\]")
        s = footref_pattern.sub(lambda m: store.add("FOOTREF", m.group(0)), s)

        # 8) 内联 HTML（保守起见冻结；若想翻译其中文本，可做白名单标签）
        inline_html_pattern = re.compile(
            r"<[A-Za-z][^>]*?>.*?</[A-Za-z][^>]*?>", re.DOTALL
        )
        s = inline_html_pattern.sub(lambda m: store.add("HTMLINLINE", m.group(0)), s)

        return s


class MarkdownTranslator:

    def __init__(self, cfg: TranslateConfig, translator_configs: dict = {}) -> None:
        self.cfg = cfg
        self.protector = MarkdownProtector()
        self.translator = self._create_translator(translator_configs)

    def _create_translator(self, translator_configs: dict = {}) -> BaseTranslator:
        if self.cfg.translator_type == "echo":
            return EchoTranslator(translator_configs)
        elif self.cfg.translator_type == "ollama":
            return OllamaTranslator(translator_configs)
        else:
            raise ValueError(f"Unsupported translator type: {self.cfg.translator_type}")

    def _split_to_nodes(self, text: str) -> dict[int, Node]:

        lines = text.splitlines(keepends=False)

        blocks: List[str] = []
        cur: List[str] = []

        def flush_block():
            if cur:
                blocks.append("\n".join(cur).rstrip("\n"))
                cur.clear()

        prev_blank = True
        for line in lines:
            # 强制在标题或新列表项前断开
            if MarkdownProtector._looks_like_heading(
                line
            ) or MarkdownProtector._looks_like_list_item(line):
                flush_block()
                cur.append(line)
                flush_block()
                prev_blank = False
                continue
            if MarkdownProtector._is_blank(line):
                cur.append("")
                flush_block()
                prev_blank = True
            else:
                cur.append(line)
                prev_blank = False
        flush_block()

        # 二次切分：过长的块按句子边界拆
        nodes: List[Node] = []
        blk_id = 0
        for blk in blocks:
            if len(blk) <= self.cfg.max_chunk_chars:
                nodes.append(Node(nid=blk_id, origin_text=blk, translated_text=""))
                blk_id += 1
            else:
                # 尝试按句号/标点切
                parts = re.split(r"(?<=[。！？!?\.])\s+", blk)
                buf = ""
                for p in parts:
                    add = (p if buf == "" else buf + " " + p).strip()
                    if len(add) > self.cfg.max_chunk_chars and buf:
                        nodes.append(
                            Node(nid=blk_id, origin_text=buf, translated_text="")
                        )
                        blk_id += 1
                        buf = p
                    else:
                        buf = add
                if buf:
                    nodes.append(Node(nid=blk_id, origin_text=buf, translated_text=""))
                    blk_id += 1

        nodes = [node for node in nodes if len(node.origin_text) > 0]

        nodes_dict = {i: nodes[i] for i in range(len(nodes))}
        return nodes_dict

    def __group_nodes(self, nodes: dict[int, Node]) -> List[str]:

        groups: List[str] = []

        cur_group = ""
        for id in nodes.keys():
            id_str = f"{id:04d}"
            node_str = f"<NODE_START_{id_str}>\n{nodes[id].origin_text}\n</NODE_END_{id_str}>\n\n"
            if len(cur_group) + len(node_str) > self.cfg.max_chunk_chars:
                groups.append(cur_group)
                cur_group = ""
            cur_group += node_str
        if cur_group:
            groups.append(cur_group)

        return groups

    def __translate_groups(self, groups: List[str]) -> List[str]:
        # 读取并发度，非法值/None 兜底为 1
        max_workers = int(getattr(self.cfg, "parallel_groups", 1) or 1)
        max_workers = max(1, max_workers)

        # 串行退化路径（便于 debug 或限流）
        if max_workers == 1 or len(groups) <= 1:
            translated_groups: List[str] = []
            for i, group in enumerate(groups):
                print(
                    f"translate.... group {i + 1}/{len(groups)} length is {len(group)}"
                )
                start_time = time.perf_counter()
                out = self.translator.translate_with_retry(
                    group, self.cfg.source_lang, self.cfg.target_lang
                )
                cost_ms = (time.perf_counter() - start_time) * 1000.0
                print(
                    f"translated group {i + 1}/{len(groups)} output length is {len(out)} time cost: {cost_ms:.3f}ms"
                )
                translated_groups.append(out)
            return translated_groups

        # 并发路径：线程池（I/O 场景优于进程池；保持保序输出）
        def worker(idx: int, group: str):
            start = time.perf_counter()
            out = self.translator.translate_with_retry(
                group, self.cfg.source_lang, self.cfg.target_lang
            )
            cost_ms = (time.perf_counter() - start) * 1000.0
            return idx, out, cost_ms

        n = len(groups)
        results: List[str] = [None] * n  # 预留，按 idx 回填，保证保序
        print(f"[parallel] launching {n} tasks with max_workers={max_workers}")

        # 可选：防止一次性提交过多任务（极大 n 时），分批提交
        batch = max_workers * 4  # 提交窗口，可按需调
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="translate"
        ) as ex:
            for start_i in range(0, n, batch):
                end_i = min(start_i + batch, n)
                futs = [ex.submit(worker, i, groups[i]) for i in range(start_i, end_i)]
                for fut in as_completed(futs):
                    idx, out, cost_ms = (
                        fut.result()
                    )  # 若 translate_with_retry 已含重试/429处理，这里可直接取
                    print(
                        f"translated group {idx + 1}/{n} output length is {len(out)} time cost: {cost_ms:.3f}ms"
                    )
                    results[idx] = out

        # 兜底校验
        missing = [i for i, v in enumerate(results) if v is None]
        if missing:
            raise RuntimeError(f"Missing translated results for indices: {missing}")

        return results

    def _ungroup_nodes(
        self, group_text: str, origin_nodes: dict[int, Node]
    ) -> dict[int, Node]:

        nodes: dict[int, Node] = {}

        # print(group_text)

        pat = re.compile(r"<NODE_START_(\d{4})>\s*(.*?)\s*</NODE_END_\1>", re.DOTALL)

        for m in pat.finditer(group_text):
            node_id = int(m.group(1))
            # print(f"find node_id {node_id}, str: {m.group(2)}")
            try:
                node_text = origin_nodes[node_id].origin_text
                nodes[node_id] = Node(
                    nid=node_id, origin_text=node_text, translated_text=m.group(2)
                )
            except Exception as e:
                print(f"Error: {node_id} not exists in origin_nodes, {e}")

        return nodes

    def _ungroup_groups(
        self, groups: List[str], origin_nodes: dict[int, Node]
    ) -> dict[int, Node]:
        nodes: dict[int, Node] = {}

        for gid in range(len(groups)):
            ungrouped_nodes = self._ungroup_nodes(groups[gid], origin_nodes)
            # print(f"group {gid} has {len(ungrouped_nodes)} nodes")
            nodes.update(ungrouped_nodes)

        print(f"ungroup start, has {len(nodes)} nodes")

        for id in origin_nodes.keys():
            if id not in nodes.keys():
                print(f"find missing {id} adding....")
                nodes[id] = origin_nodes[id]

        return nodes

    def _collect_failed_nodes(self, nodes: dict[int, Node]) -> dict[int, Node]:
        failed_nodes: dict[int, Node] = {}
        for id in nodes.keys():
            if nodes[id].translated_text == "":
                # print(f"failed node {id} length is {len(nodes[id].translated_text)}")
                failed_nodes[id] = nodes[id]
        return failed_nodes

    def merge_nodes(self, nodes: dict[int, Node]) -> str:
        text = ""
        # sort nodes by id
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
        for id, node in sorted_nodes:
            text += f"\n\n{node.translated_text}\n\n"
        return text

    def translate(
        self, text: str
    ) -> tuple[str, str, PlaceHolderStore, dict[int, Node]]:

        store = PlaceHolderStore()

        protected_text = self.protector.protect(text, self.cfg, store)

        nodes = self._split_to_nodes(protected_text)

        print(f"Has {len(nodes)} nodes")

        groups = self.__group_nodes(nodes)

        print(f"Has {len(groups)} groups")

        groups = groups

        # for gid in range(len(groups)):
        #     print(f"group {gid} length is {len(groups[gid])}")

        translated_groups = self.__translate_groups(groups)

        translated_nodes = self._ungroup_groups(translated_groups, nodes)

        print(f"Has {len(translated_nodes)} translated nodes")

        failed_nodes = self._collect_failed_nodes(translated_nodes)

        print(f"Has {len(failed_nodes)} failed nodes")

        # for id in failed_nodes.keys():
        #     print(
        #         f"failed node {id} length is {len(failed_nodes[id].origin_text)}, origin text: {nodes[id].origin_text}"
        #     )

        translated_text = self.merge_nodes(translated_nodes)

        translated_text = self.protector.unprotect(translated_text, store)

        translated_text = remove_unnecessary_whitespaces(translated_text)

        # print(translated_text)

        return translated_text, protected_text, store, translated_nodes


def nodes_to_dict(nodes: dict[int, Node]) -> dict:
    result = {}
    for id, node in nodes.items():
        result[id] = {}
        node_dict = node.__dict__
        for key, value in node_dict.items():
            if isinstance(value, str):
                # fix encoding
                result[id][key] = value.encode("utf-8").decode("utf-8")
            elif isinstance(value, list):
                result[id][key] = [v.__dict__ for v in value]
            elif isinstance(value, dict):
                result[id][key] = {k: v.__dict__ for k, v in value.items()}
            else:
                result[id][key] = value
    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str, default="input.md", help="markdown file to translate"
    )
    parser.add_argument(
        "output_file", type=str, default="output.md", help="translated markdown file"
    )
    parser.add_argument(
        "-s", "--source-lang", type=str, default="en", help="source language"
    )
    parser.add_argument(
        "-t", "--target-lang", type=str, default="zh", help="target language"
    )
    parser.add_argument(
        "-d",
        "--protected-database",
        type=str,
        default="protected.json",
        help="protected map json",
    )
    parser.add_argument(
        "-p",
        "--protected-file",
        type=str,
        default="protected.md",
        help="protected map markdown",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="config file",
    )
    parser.add_argument(
        "-n",
        "--nodes-file",
        type=str,
        default="nodes.json",
        help="nodes file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: input file {args.input_file} does not exist")
        exit(1)

    if args.config:
        cfg, translator_cfg = load_configs(args.config)
    else:
        cfg = create_default_cfg()
        translator_cfg = {}

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    translator = MarkdownTranslator(cfg, translator_cfg)

    with open(args.input_file, "r") as f:
        text = f.read()

    translated_text, protected_text, store, final_nodes = translator.translate(text)

    if args.protected_file:
        with open(args.protected_file, "w") as f:
            f.write(protected_text)

    if args.protected_database:
        store.save(args.protected_database)

    if args.nodes_file:
        final_nodes_json = nodes_to_dict(final_nodes)
        with open(args.nodes_file, "w") as f:
            json.dump(final_nodes_json, f, indent=4, ensure_ascii=False)

    with open(args.output_file, "w") as f:
        f.write(translated_text)
