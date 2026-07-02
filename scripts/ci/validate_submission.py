#!/usr/bin/env python3
"""
M5 最终提交物一键打包与版本校验脚本

功能：
1. --check 模式：校验所有提交物是否符合清单要求
2. --pack 模式：校验 + 创建最终提交压缩包

作者：量子RL调度系统团队
日期：2026-07-02
"""

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class SubmissionValidator:
    """提交物校验器"""

    def __init__(self, manifest_path: Path, project_root: Path):
        """
        初始化校验器

        Args:
            manifest_path: 清单文件路径
            project_root: 项目根目录
        """
        self.manifest_path = manifest_path
        self.project_root = project_root
        self.manifest = self._load_manifest()
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _load_manifest(self) -> dict[str, Any]:
        """加载清单文件"""
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"无法加载清单文件: {e}") from e

    def validate_all(self) -> bool:
        """
        校验所有提交物

        Returns:
            True: 所有校验通过
            False: 存在错误
        """
        print("=" * 80)
        print("开始校验最终提交物")
        print("=" * 80)

        submission_info = self.manifest.get("submission", {})
        version = submission_info.get("version", "unknown")
        deadline = submission_info.get("deadline", "unknown")

        print(f"\n版本: {version}")
        print(f"截止日期: {deadline}")
        print()

        items = self.manifest.get("items", [])

        for item in items:
            self._validate_item(item)

        # 打印结果
        print("\n" + "=" * 80)
        print("校验结果")
        print("=" * 80)

        if self.errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ 所有校验通过！")

        return len(self.errors) == 0

    def _validate_item(self, item: dict[str, Any]) -> None:
        """
        校验单个提交物

        Args:
            item: 提交物定义
        """
        item_id = item.get("id", "UNKNOWN")
        item_name = item.get("name", "未命名")
        item_type = item.get("type", "unknown")
        item_path = item.get("path", "")

        print(f"\n[{item_id}] {item_name}")

        # 特殊处理 git_tag 类型
        if item_type == "git_tag":
            self._validate_git_tag(item)
            return

        # 特殊处理 zip 类型（代码压缩包）
        if item_type == "zip":
            self._validate_zip(item)
            return

        # 检查文件是否存在
        full_path = self.project_root / item_path
        if not full_path.exists():
            self.errors.append(f"{item_id}: 文件不存在 - {item_path}")
            print(f"  ❌ 文件不存在: {item_path}")
            return

        print(f"  ✓ 文件存在: {item_path}")

        # 根据类型进行特定校验
        if item_type == "pdf":
            self._validate_pdf(item, full_path)
        elif item_type == "pptx":
            self._validate_pptx(item, full_path)
        elif item_type == "mp4":
            self._validate_mp4(item, full_path)
        elif item_type == "md":
            # Markdown 文件只需检查存在性
            pass
        else:
            self.warnings.append(f"{item_id}: 未知类型 - {item_type}")

    def _validate_git_tag(self, item: dict[str, Any]) -> None:
        """校验 Git 标签"""
        item_id = item.get("id", "UNKNOWN")
        requirements = item.get("requirements", {})
        tag = requirements.get("tag", "")

        print(f"  检查 Git 标签: {tag}")

        try:
            # 检查标签是否存在
            result = subprocess.run(
                ["git", "tag", "-l", tag],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            if tag in result.stdout:
                print(f"  ✓ Git 标签存在: {tag}")
            else:
                self.errors.append(f"{item_id}: Git 标签不存在 - {tag}")
                print(f"  ❌ Git 标签不存在: {tag}")

        except subprocess.CalledProcessError as e:
            self.warnings.append(f"{item_id}: 无法检查 Git 标签 - {e}")
            print(f"  ⚠️  无法检查 Git 标签: {e}")

    def _validate_zip(self, item: dict[str, Any]) -> None:
        """校验 ZIP 压缩包"""
        item_id = item.get("id", "UNKNOWN")
        item_path = item.get("path", "")
        requirements = item.get("requirements", {})
        max_size_mb = requirements.get("max_size_mb", 100)

        full_path = self.project_root / item_path

        # 如果压缩包不存在，这是预期的（将在 pack 模式创建）
        if not full_path.exists():
            print(f"  ℹ️  压缩包不存在（将在 pack 模式创建）: {item_path}")
            return

        # 检查大小
        size_mb = full_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            self.errors.append(
                f"{item_id}: 压缩包过大 - {size_mb:.2f}MB > {max_size_mb}MB"
            )
            print(f"  ❌ 压缩包过大: {size_mb:.2f}MB > {max_size_mb}MB")
        else:
            print(f"  ✓ 压缩包大小: {size_mb:.2f}MB <= {max_size_mb}MB")

    def _validate_pdf(self, item: dict[str, Any], pdf_path: Path) -> None:
        """校验 PDF 文件"""
        item_id = item.get("id", "UNKNOWN")
        requirements = item.get("requirements", {})
        min_pages = requirements.get("min_pages", 0)
        max_pages = requirements.get("max_pages", 1000)
        must_contain = requirements.get("must_contain", [])

        # 尝试使用 PyPDF2
        try:
            import PyPDF2

            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

                if num_pages < min_pages:
                    self.errors.append(
                        f"{item_id}: PDF 页数过少 - {num_pages} < {min_pages}"
                    )
                    print(f"  ❌ PDF 页数过少: {num_pages} < {min_pages}")
                elif num_pages > max_pages:
                    self.errors.append(
                        f"{item_id}: PDF 页数过多 - {num_pages} > {max_pages}"
                    )
                    print(f"  ❌ PDF 页数过多: {num_pages} > {max_pages}")
                else:
                    print(f"  ✓ PDF 页数: {num_pages}")

                # 检查必须包含的内容
                if must_contain:
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""

                    for keyword in must_contain:
                        if keyword in text:
                            print(f"  ✓ 包含关键词: {keyword}")
                        else:
                            self.errors.append(
                                f"{item_id}: PDF 缺少关键词 - {keyword}"
                            )
                            print(f"  ❌ 缺少关键词: {keyword}")

        except ImportError:
            self.warnings.append(
                f"{item_id}: PyPDF2 未安装，跳过 PDF 详细校验"
            )
            print("  ⚠️  PyPDF2 未安装，跳过详细校验")
        except Exception as e:
            self.errors.append(f"{item_id}: PDF 校验失败 - {e}")
            print(f"  ❌ PDF 校验失败: {e}")

    def _validate_pptx(self, item: dict[str, Any], pptx_path: Path) -> None:
        """校验 PPTX 文件"""
        item_id = item.get("id", "UNKNOWN")
        requirements = item.get("requirements", {})
        min_slides = requirements.get("min_slides", 0)
        max_slides = requirements.get("max_slides", 100)
        must_contain_slides = requirements.get("must_contain_slides", [])

        # 尝试使用 python-pptx
        try:
            from pptx import Presentation

            prs = Presentation(pptx_path)
            num_slides = len(prs.slides)

            if num_slides < min_slides:
                self.errors.append(
                    f"{item_id}: PPT 幻灯片过少 - {num_slides} < {min_slides}"
                )
                print(f"  ❌ PPT 幻灯片过少: {num_slides} < {min_slides}")
            elif num_slides > max_slides:
                self.errors.append(
                    f"{item_id}: PPT 幻灯片过多 - {num_slides} > {max_slides}"
                )
                print(f"  ❌ PPT 幻灯片过多: {num_slides} > {max_slides}")
            else:
                print(f"  ✓ PPT 幻灯片数: {num_slides}")

            # 检查必须包含的幻灯片（简单检查标题）
            if must_contain_slides:
                slide_titles = []
                for slide in prs.slides:
                    if slide.shapes.title:
                        slide_titles.append(slide.shapes.title.text)

                for required_title in must_contain_slides:
                    found = any(
                        required_title in title for title in slide_titles
                    )
                    if found:
                        print(f"  ✓ 包含幻灯片: {required_title}")
                    else:
                        self.warnings.append(
                            f"{item_id}: 未找到幻灯片 - {required_title}"
                        )
                        print(f"  ⚠️  未找到幻灯片: {required_title}")

        except ImportError:
            self.warnings.append(
                f"{item_id}: python-pptx 未安装，跳过 PPTX 详细校验"
            )
            print("  ⚠️  python-pptx 未安装，跳过详细校验")
        except Exception as e:
            self.errors.append(f"{item_id}: PPTX 校验失败 - {e}")
            print(f"  ❌ PPTX 校验失败: {e}")

    def _validate_mp4(self, item: dict[str, Any], mp4_path: Path) -> None:
        """校验 MP4 视频文件"""
        item_id = item.get("id", "UNKNOWN")
        requirements = item.get("requirements", {})
        max_duration = requirements.get("max_duration_seconds", 300)
        min_duration = requirements.get("min_duration_seconds", 0)
        max_size_mb = requirements.get("max_size_mb", 500)
        expected_resolution = requirements.get("resolution", "1920x1080")

        # 检查文件大小
        size_mb = mp4_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            self.errors.append(
                f"{item_id}: 视频文件过大 - {size_mb:.2f}MB > {max_size_mb}MB"
            )
            print(f"  ❌ 视频文件过大: {size_mb:.2f}MB > {max_size_mb}MB")
        else:
            print(f"  ✓ 视频文件大小: {size_mb:.2f}MB")

        # 尝试使用 ffprobe 检查时长和分辨率
        try:
            # 获取时长
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-showentries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(mp4_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            duration = float(result.stdout.strip())
            if duration < min_duration:
                self.errors.append(
                    f"{item_id}: 视频时长过短 - {duration:.1f}s < {min_duration}s"
                )
                print(f"  ❌ 视频时长过短: {duration:.1f}s < {min_duration}s")
            elif duration > max_duration:
                self.errors.append(
                    f"{item_id}: 视频时长过长 - {duration:.1f}s > {max_duration}s"
                )
                print(f"  ❌ 视频时长过长: {duration:.1f}s > {max_duration}s")
            else:
                print(f"  ✓ 视频时长: {duration:.1f}s")

            # 获取分辨率
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-showentries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    str(mp4_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            resolution = result.stdout.strip()
            if resolution != expected_resolution:
                self.warnings.append(
                    f"{item_id}: 视频分辨率不匹配 - {resolution} != {expected_resolution}"
                )
                print(
                    f"  ⚠️  视频分辨率: {resolution} (期望: {expected_resolution})"
                )
            else:
                print(f"  ✓ 视频分辨率: {resolution}")

        except FileNotFoundError:
            self.warnings.append(
                f"{item_id}: ffprobe 未安装，跳过视频时长和分辨率校验"
            )
            print("  ⚠️  ffprobe 未安装，跳过时长和分辨率校验")
        except subprocess.CalledProcessError as e:
            self.warnings.append(
                f"{item_id}: ffprobe 执行失败 - {e}"
            )
            print(f"  ⚠️  ffprobe 执行失败: {e}")
        except Exception as e:
            self.errors.append(f"{item_id}: 视频校验失败 - {e}")
            print(f"  ❌ 视频校验失败: {e}")

    def pack_submission(self, output_dir: Path) -> bool:
        """
        打包提交物

        Args:
            output_dir: 输出目录

        Returns:
            True: 打包成功
            False: 打包失败
        """
        # 先校验
        if not self.validate_all():
            print("\n❌ 校验失败，无法打包")
            return False

        print("\n" + "=" * 80)
        print("开始打包提交物")
        print("=" * 80)

        submission_info = self.manifest.get("submission", {})
        version = submission_info.get("version", "unknown")
        date_str = datetime.now().strftime("%Y%m%d")

        # 创建输出文件名
        output_filename = f"submission_v{version}_{date_str}.zip"
        output_path = output_dir / output_filename

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取需要打包的文件
        items = self.manifest.get("items", [])
        files_to_pack = []

        for item in items:
            item_type = item.get("type", "")
            item_path = item.get("path", "")

            # 跳过 git_tag 类型
            if item_type == "git_tag":
                continue

            # 跳过 CODE_ARCHIVE（这是我们要创建的）
            if item.get("id") == "CODE_ARCHIVE":
                continue

            full_path = self.project_root / item_path
            if full_path.exists() and full_path.is_file():
                files_to_pack.append((full_path, item_path))

        # 创建 ZIP 文件
        try:
            with zipfile.ZipFile(
                output_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                for file_path, arcname in files_to_pack:
                    zipf.write(file_path, arcname)
                    print(f"  ✓ 添加: {arcname}")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ 打包完成: {output_path}")
            print(f"   大小: {size_mb:.2f}MB")
            print(f"   包含 {len(files_to_pack)} 个文件")

            return True

        except Exception as e:
            print(f"\n❌ 打包失败: {e}")
            return False


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="M5 最终提交物一键打包与版本校验脚本"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/submission_manifest.yaml"),
        help="清单文件路径（默认: config/submission_manifest.yaml）",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录（默认: 当前目录）",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="输出目录（默认: dist）",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="仅校验模式",
    )
    mode_group.add_argument(
        "--pack",
        action="store_true",
        help="校验并打包模式",
    )

    args = parser.parse_args()

    # 检查清单文件
    if not args.manifest.exists():
        print(f"❌ 清单文件不存在: {args.manifest}")
        return 1

    # 创建校验器
    try:
        validator = SubmissionValidator(args.manifest, args.project_root)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return 1

    # 执行操作
    if args.check:
        success = validator.validate_all()
        return 0 if success else 1
    elif args.pack:
        success = validator.pack_submission(args.output_dir)
        return 0 if success else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
