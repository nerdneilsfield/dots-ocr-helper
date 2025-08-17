#!/usr/bin/env python3

"""
PDF Pages to Complete Markdown Converter
将 JSONL 格式的 PDF 页面信息转换为完整的 Markdown 文档
处理原始 markdown 中的 base64 内嵌图片
"""

import argparse
import base64
import logging
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import coloredlogs

    coloredlogs.install(level=logging.INFO)
except ImportError:
    pass


def _decode_and_save_image_to_dir(
    base64_data: str, page_no: int, image_index: int, images_dir: str
) -> Optional[str]:
    """在独立函数中进行解码与写入，便于进程池调用。

    Returns:
        相对路径字符串，例如 "images/xxx.png"；失败返回 None。
    """
    try:
        image_data = base64.b64decode(base64_data)

        ext = ".png"
        if image_data[:2] == b"\xff\xd8":
            ext = ".jpg"
        elif image_data[:3] == b"GIF":
            ext = ".gif"

        filename = f"page_{page_no:03d}_img_{image_index:03d}{ext}"
        filepath = Path(images_dir) / filename
        with open(filepath, "wb") as f:
            f.write(image_data)

        return f"images/{filename}"
    except Exception as e:
        logging.error(f"Error saving image (mp): {e}")
        return None


class PDFToMarkdownConverter:
    def __init__(
        self,
        jsonl_path: str,
        output_dir: str,
        concurrency: str = "none",
        max_workers: Optional[int] = None,
    ):
        """
        初始化转换器

        Args:
            jsonl_path: 输入的 JSONL 文件路径
            output_dir: 输出目录路径
        """
        self.jsonl_path = Path(jsonl_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.title = "_".join(os.path.basename(jsonl_path).split(".")[:-1])

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # 并发设置
        concurrency = concurrency.lower().strip()
        if concurrency not in {"none", "thread", "process"}:
            raise ValueError("concurrency must be one of: none, thread, process")
        self.concurrency = concurrency
        self.max_workers = (
            max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4)
        )

        # 存储所有页面数据
        self.pages_data: List[Dict] = []

        # 用于跟踪提取的图片
        self.image_counter = 0

    def load_jsonl(self) -> None:
        """加载 JSONL 文件"""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.pages_data.append(json.loads(line))

        # 按页码排序
        self.pages_data.sort(key=lambda x: x.get("page_no", 0))
        logging.info(f"Loaded {len(self.pages_data)} pages from JSONL")

    def extract_base64_images(self, markdown_content: str) -> List[Tuple[str, str]]:
        """
        从 markdown 内容中提取 base64 图片

        Returns:
            List of tuples: (original_markdown_image, base64_data)
        """
        # 匹配 markdown 中的 base64 图片
        pattern = r"!\[([^\]]*)\]\((data:image/[^;]+;base64,([^)]+))\)"
        matches = re.findall(pattern, markdown_content)

        images = []
        for match in matches:
            alt_text = match[0]
            full_data_url = match[1]
            base64_data = match[2]
            original = f"![{alt_text}]({full_data_url})"
            images.append((original, base64_data))

        return images

    def save_base64_as_file(
        self, base64_data: str, page_no: int, image_index: Optional[int] = None
    ) -> Optional[str]:
        """
        将 base64 数据保存为图片文件

        Returns:
            相对路径字符串
        """
        try:
            # 解码 base64 数据
            image_data = base64.b64decode(base64_data)

            # 尝试检测图片格式（简单检测）
            ext = ".png"  # 默认 PNG
            if image_data[:2] == b"\xff\xd8":
                ext = ".jpg"
            elif image_data[:3] == b"GIF":
                ext = ".gif"

            # 生成文件名
            if image_index is None:
                self.image_counter += 1
                index_for_name = self.image_counter
            else:
                index_for_name = image_index
            filename = f"page_{page_no:03d}_img_{index_for_name:03d}{ext}"
            filepath = self.images_dir / filename

            # 保存文件
            with open(filepath, "wb") as f:
                f.write(image_data)

            return f"images/{filename}"
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return None

    def read_markdown_content(self, page_data: Dict, use_nohf: bool = False) -> str:
        """读取页面的 Markdown 内容"""
        # 选择使用哪个 markdown 文件
        md_key = "md_content_nohf_path" if use_nohf else "md_content_path"
        md_path = page_data.get(md_key)

        if not md_path or not os.path.exists(md_path):
            logging.warning(
                f"Warning: Markdown file not found for page {page_data.get('page_no', 'unknown')}"
            )
            return ""

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading markdown file {md_path}: {e}")
            return ""

    def process_markdown_with_external_images(self, content: str, page_no: int) -> str:
        """
        处理 markdown 内容，将 base64 图片转换为外部文件引用
        """
        if not content:
            return content

        # 提取所有 base64 图片
        images = self.extract_base64_images(content)

        logging.info(f"Page {page_no} has {len(images)} images")

        # 替换每个图片
        processed_content = content

        # 按需并发保存图片
        if self.concurrency == "none" or len(images) <= 1:
            for idx, (original, base64_data) in enumerate(images, start=1):
                relative_path = self.save_base64_as_file(
                    base64_data, page_no, image_index=idx
                )
                if relative_path:
                    alt_match = re.match(r"!\[([^\]]*)\]", original)
                    alt_text = alt_match.group(1) if alt_match else ""
                    new_image = f"![{alt_text}]({relative_path})"
                    processed_content = processed_content.replace(original, new_image)
        else:
            # 构造任务列表
            tasks = []
            originals: List[str] = []
            for idx, (original, base64_data) in enumerate(images, start=1):
                originals.append(original)
                tasks.append((base64_data, idx))

            replacement_map: Dict[str, str] = {}

            if self.concurrency == "thread":
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_original = {
                        executor.submit(
                            self.save_base64_as_file, base64_data, page_no, idx
                        ): original
                        for (base64_data, idx), original in zip(tasks, originals)
                    }
                    for future in as_completed(future_to_original):
                        original = future_to_original[future]
                        relative_path = future.result()
                        if relative_path:
                            replacement_map[original] = relative_path
            else:  # process
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_original = {
                        executor.submit(
                            _decode_and_save_image_to_dir,
                            base64_data,
                            page_no,
                            idx,
                            str(self.images_dir),
                        ): original
                        for (base64_data, idx), original in zip(tasks, originals)
                    }
                    for future in as_completed(future_to_original):
                        original = future_to_original[future]
                        relative_path = future.result()
                        if relative_path:
                            replacement_map[original] = relative_path

            for original, relative_path in replacement_map.items():
                alt_match = re.match(r"!\[([^\]]*)\]", original)
                alt_text = alt_match.group(1) if alt_match else ""
                new_image = f"![{alt_text}]({relative_path})"
                processed_content = processed_content.replace(original, new_image)

        return processed_content

    def generate_markdown_with_base64(
        self, use_nohf: bool = False, add_page_title: bool = False
    ) -> str:
        """生成包含 base64 内嵌图片的 Markdown（保留原始格式）"""
        markdown_parts = []

        # 添加文档标题
        pdf_name = (
            Path(self.pages_data[0].get("file_path", "Unknown")).stem
            if self.pages_data
            else "Unknown"
        )
        # markdown_parts.append(f"# {pdf_name}\n\n")
        # markdown_parts.append(
        #     f"*Generated from PDF with {len(self.pages_data)} pages*\n\n"
        # )
        # markdown_parts.append("---\n\n")

        for page_data in self.pages_data:
            page_no = page_data.get("page_no", 0)

            # 添加页面标题
            if add_page_title:
                markdown_parts.append(f"## Page {page_no + 1}\n\n")

            # 添加页面内容（保留原始的 base64 图片）
            content = self.read_markdown_content(page_data, use_nohf)
            if content:
                markdown_parts.append(content)
                markdown_parts.append("\n\n")

            # 添加页面分隔符
            markdown_parts.append("---\n\n")

        return "".join(markdown_parts)

    def generate_markdown_with_external_images(
        self, use_nohf: bool = False, add_page_title: bool = False
    ) -> str:
        """生成使用外部图片链接的 Markdown"""
        markdown_parts = []
        self.image_counter = 0  # 重置图片计数器

        # 添加文档标题
        pdf_name = (
            Path(self.pages_data[0].get("file_path", "Unknown")).stem
            if self.pages_data
            else "Unknown"
        )
        markdown_parts.append(f"# {pdf_name}\n\n")
        markdown_parts.append(
            f"*Generated from PDF with {len(self.pages_data)} pages*\n\n"
        )
        markdown_parts.append("---\n\n")

        for page_data in self.pages_data:
            page_no = page_data.get("page_no", 0)

            # 添加页面标题
            if add_page_title:
                markdown_parts.append(f"## Page {page_no + 1}\n\n")

            # 处理页面内容，转换 base64 图片为外部文件
            content = self.read_markdown_content(page_data, use_nohf)
            if content:
                processed_content = self.process_markdown_with_external_images(
                    content, page_no
                )
                markdown_parts.append(processed_content)
                markdown_parts.append("\n\n")

            # 添加页面分隔符
            markdown_parts.append("---\n\n")

        return "".join(markdown_parts)

    def save_markdown(self, content: str, filename: str) -> None:
        """保存 Markdown 文件"""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Saved: {output_path}")

    def generate_metadata(self) -> None:
        """生成元数据文件"""
        metadata = {
            "total_pages": len(self.pages_data),
            "source_pdf": (
                self.pages_data[0].get("file_path", "Unknown")
                if self.pages_data
                else "Unknown"
            ),
            "pages": [],
        }

        for page_data in self.pages_data:
            # 读取内容以统计图片数量
            content = self.read_markdown_content(page_data)
            images = self.extract_base64_images(content)

            page_info = {
                "page_no": page_data.get("page_no", 0),
                "has_content": bool(page_data.get("md_content_path")),
                "has_content_nohf": bool(page_data.get("md_content_nohf_path")),
                "embedded_images": len(images),
                "input_dimensions": {
                    "width": page_data.get("input_width", 0),
                    "height": page_data.get("input_height", 0),
                },
            }
            metadata["pages"].append(page_info)

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved metadata: {metadata_path}")

    def convert(self, use_nohf: bool = False, add_page_title: bool = False) -> None:
        """执行转换过程"""
        logging.info(f"Starting conversion...")
        logging.info(f"Input: {self.jsonl_path}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(
            f"Using {'no header/footer' if use_nohf else 'full'} markdown content\n"
        )

        # 加载数据
        self.load_jsonl()

        if not self.pages_data:
            logging.info("No pages found in JSONL file")
            return

        # 生成包含 base64 图片的 Markdown（保留原始格式）
        logging.info("Generating Markdown with embedded base64 images...")
        md_base64 = self.generate_markdown_with_base64(use_nohf, add_page_title)
        self.save_markdown(md_base64, f"{self.title}_embedded.md")

        # 生成使用外部图片的 Markdown
        logging.info("\nGenerating Markdown with external images...")
        md_external = self.generate_markdown_with_external_images(
            use_nohf, add_page_title
        )
        self.save_markdown(md_external, f"{self.title}_external.md")

        # 生成元数据
        logging.info("\nGenerating metadata...")
        self.generate_metadata()

        # 统计提取的图片数量
        image_files = list(self.images_dir.glob("*"))
        image_count = len([f for f in image_files if f.is_file()])

        logging.info("\n✅ Conversion completed successfully!")
        logging.info(f"Output files:")
        logging.info(f"  - {self.output_dir}/{self.title}_embedded.md")
        logging.info(f"  - {self.output_dir}/{self.title}_external.md")
        logging.info(
            f"  - {self.output_dir}/images/ (containing {image_count} extracted images)"
        )
        logging.info(f"  - {self.output_dir}/metadata.json")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Convert JSONL PDF pages data to complete Markdown documents"
    )
    parser.add_argument("jsonl_path", help="Path to the input JSONL file")
    parser.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "-n",
        "--no-header-footer",
        action="store_true",
        help="Use markdown content without headers/footers",
    )
    parser.add_argument(
        "-p", "--page", action="store_true", help="Add page number as a subtitle title "
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        choices=["none", "thread", "process"],
        default="none",
        help="Concurrency mode for image extraction (none/thread/process)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Max workers for concurrency (default: CPU count)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
    )

    # 创建转换器并执行
    converter = PDFToMarkdownConverter(
        args.jsonl_path,
        args.output,
        concurrency=args.concurrency,
        max_workers=args.workers,
    )
    converter.convert(use_nohf=args.no_header_footer, add_page_title=args.page)


if __name__ == "__main__":
    main()
