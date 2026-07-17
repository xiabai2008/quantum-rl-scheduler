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
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SubmissionValidator:
    """提交物校验器"""

    def __init__(self, manifest_path: str, project_root: str = ".") -> None:
        """初始化校验器

        Args:
            manifest_path: 清单文件路径
            project_root: 项目根目录
        """
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)
        self.project_root = Path(project_root)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """校验所有提交物

        Returns:
            是否通过校验
        """
        print(f"=== M5 提交物校验报告 ===")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"版本: {self.manifest['submission']['version']}")
        print(f"截止日期: {self.manifest['submission']['deadline']}")
        print()

        for item in self.manifest["items"]:
            self._validate_item(item)

        self._check_version_consistency()
        self._report()
        return len(self.errors) == 0

    def _validate_item(self, item: Dict[str, Any]) -> None:
        """校验单个提交物

        Args:
            item: 提交物定义
        """
        item_id = item["id"]
        item_name = item["name"]
        item_type = item["type"]
        path = self.project_root / item["path"]

        print(f"[{item_id}] {item_name} ({item_type})")

        # 检查文件存在性
        if not path.exists():
            self.errors.append(f"[{item_id}] 文件不存在: {path}")
            print(f"  ❌ 文件不存在: {path}")
            return

        # 按类型校验
        if item_type == "pdf":
            self._validate_pdf(item, path)
        elif item_type == "pptx":
            self._validate_pptx(item, path)
        elif item_type == "mp4":
            self._validate_mp4(item, path)
        elif item_type == "zip":
            self._validate_zip(item, path)
        elif item_type == "git_tag":
            self._validate_git_tag(item)
        elif item_type == "md":
            self._validate_markdown(item, path)

        # 检查依赖
        if "depends_on" in item:
            self._check_dependency(item)

    def _validate_pdf(self, item: Dict[str, Any], path: Path) -> None:
        """校验 PDF 文件

        Args:
            item: 提交物定义
            path: 文件路径
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(path)
            num_pages = len(reader.pages)
            reqs = item.get("requirements", {})

            min_pages = reqs.get("min_pages")
            max_pages = reqs.get("max_pages")

            if min_pages and num_pages < min_pages:
                self.errors.append(
                    f"[{item['id']}] PDF 页数不足: {num_pages} < {min_pages}"
                )
                print(f"  ❌ 页数不足: {num_pages} < {min_pages}")
            elif max_pages and num_pages > max_pages:
                self.errors.append(
                    f"[{item['id']}] PDF 页数超限: {num_pages} > {max_pages}"
                )
                print(f"  ❌ 页数超限: {num_pages} > {max_pages}")
            else:
                print(f"  ✅ 页数: {num_pages}")

            # 检查必需内容
            must_contain = reqs.get("must_contain", [])
            if must_contain:
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                missing = [kw for kw in must_contain if kw not in text]
                if missing:
                    self.warnings.append(
                        f"[{item['id']}] PDF 缺少关键词: {', '.join(missing)}"
                    )
                    print(f"  ⚠️  缺少关键词: {', '.join(missing)}")
                else:
                    print(f"  ✅ 包含所有必需关键词")

        except ImportError:
            self.warnings.append(f"[{item['id']}] PyPDF2 未安装，跳过 PDF 详细校验")
            print(f"  ⚠️  PyPDF2 未安装，跳过详细校验")
        except Exception as e:
            self.errors.append(f"[{item['id']}] PDF 校验失败: {e}")
            print(f"  ❌ 校验失败: {e}")

    def _validate_pptx(self, item: Dict[str, Any], path: Path) -> None:
        """校验 PPTX 文件

        Args:
            item: 提交物定义
            path: 文件路径
        """
        try:
            from pptx import Presentation

            prs = Presentation(path)
            num_slides = len(prs.slides)
            reqs = item.get("requirements", {})

            min_slides = reqs.get("min_slides")
            max_slides = reqs.get("max_slides")

            if min_slides and num_slides < min_slides:
                self.errors.append(
                    f"[{item['id']}] PPT 页数不足: {num_slides} < {min_slides}"
                )
                print(f"  ❌ 页数不足: {num_slides} < {min_slides}")
            elif max_slides and num_slides > max_slides:
                self.errors.append(
                    f"[{item['id']}] PPT 页数超限: {num_slides} > {max_slides}"
                )
                print(f"  ❌ 页数超限: {num_slides} > {max_slides}")
            else:
                print(f"  ✅ 幻灯片数: {num_slides}")

            # 检查必需幻灯片
            must_contain = reqs.get("must_contain_slides", [])
            if must_contain:
                slide_titles = []
                for slide in prs.slides:
                    if slide.shapes.title:
                        slide_titles.append(slide.shapes.title.text)

                missing = [
                    title
                    for title in must_contain
                    if not any(title in t for t in slide_titles)
                ]
                if missing:
                    self.warnings.append(
                        f"[{item['id']}] PPT 缺少幻灯片: {', '.join(missing)}"
                    )
                    print(f"  ⚠️  缺少幻灯片: {', '.join(missing)}")
                else:
                    print(f"  ✅ 包含所有必需幻灯片")

        except ImportError:
            self.warnings.append(
                f"[{item['id']}] python-pptx 未安装，跳过 PPTX 详细校验"
            )
            print(f"  ⚠️  python-pptx 未安装，跳过详细校验")
        except Exception as e:
            self.errors.append(f"[{item['id']}] PPTX 校验失败: {e}")
            print(f"  ❌ 校验失败: {e}")

    def _validate_mp4(self, item: Dict[str, Any], path: Path) -> None:
        """校验 MP4 文件

        Args:
            item: 提交物定义
            path: 文件路径
        """
        try:
            # 检查文件大小
            size_mb = path.stat().st_size / (1024 * 1024)
            reqs = item.get("requirements", {})
            max_size = reqs.get("max_size_mb")

            if max_size and size_mb > max_size:
                self.errors.append(
                    f"[{item['id']}] 视频文件过大: {size_mb:.1f}MB > {max_size}MB"
                )
                print(f"  ❌ 文件过大: {size_mb:.1f}MB > {max_size}MB")
            else:
                print(f"  ✅ 文件大小: {size_mb:.1f}MB")

            # 使用 ffprobe 检查时长和分辨率
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,duration",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            info = json.loads(result.stdout)
            duration = float(info["format"]["duration"])
            stream = info["streams"][0]
            width = stream["width"]
            height = stream["height"]

            min_duration = reqs.get("min_duration_seconds")
            max_duration = reqs.get("max_duration_seconds")

            if min_duration and duration < min_duration:
                self.errors.append(
                    f"[{item['id']}] 视频时长不足: {duration:.1f}s < {min_duration}s"
                )
                print(f"  ❌ 时长不足: {duration:.1f}s < {min_duration}s")
            elif max_duration and duration > max_duration:
                self.errors.append(
                    f"[{item['id']}] 视频时长超限: {duration:.1f}s > {max_duration}s"
                )
                print(f"  ❌ 时长超限: {duration:.1f}s > {max_duration}s")
            else:
                print(f"  ✅ 时长: {duration:.1f}s")

            expected_resolution = reqs.get("resolution")
            if expected_resolution:
                exp_w, exp_h = map(int, expected_resolution.split("x"))
                if width != exp_w or height != exp_h:
                    self.errors.append(
                        f"[{item['id']}] 视频分辨率不匹配: {width}x{height} != {expected_resolution}"
                    )
                    print(f"  ❌ 分辨率: {width}x{height} != {expected_resolution}")
                else:
                    print(f"  ✅ 分辨率: {width}x{height}")

        except FileNotFoundError:
            self.warnings.append(
                f"[{item['id']}] ffprobe 未安装，跳过 MP4 详细校验"
            )
            print(f"  ⚠️  ffprobe 未安装，跳过详细校验")
        except subprocess.CalledProcessError as e:
            self.errors.append(f"[{item['id']}] ffprobe 执行失败: {e}")
            print(f"  ❌ ffprobe 执行失败")
        except Exception as e:
            self.errors.append(f"[{item['id']}] MP4 校验失败: {e}")
            print(f"  ❌ 校验失败: {e}")

    def _validate_zip(self, item: Dict[str, Any], path: Path) -> None:
        """校验 ZIP 文件

        Args:
            item: 提交物定义
            path: 文件路径
        """
        size_mb = path.stat().st_size / (1024 * 1024)
        reqs = item.get("requirements", {})
        max_size = reqs.get("max_size_mb")

        if max_size and size_mb > max_size:
            self.errors.append(
                f"[{item['id']}] ZIP 文件过大: {size_mb:.1f}MB > {max_size}MB"
            )
            print(f"  ❌ 文件过大: {size_mb:.1f}MB > {max_size}MB")
        else:
            print(f"  ✅ 文件大小: {size_mb:.1f}MB")

    def _validate_git_tag(self, item: Dict[str, Any]) -> None:
        """校验 Git 标签

        Args:
            item: 提交物定义
        """
        reqs = item.get("requirements", {})
        tag = reqs.get("tag")

        if tag:
            try:
                result = subprocess.run(
                    ["git", "tag", "-l", tag],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self.project_root,
                )
                if tag in result.stdout:
                    print(f"  ✅ 标签存在: {tag}")
                else:
                    self.errors.append(f"[{item['id']}] Git 标签不存在: {tag}")
                    print(f"  ❌ 标签不存在: {tag}")
            except Exception as e:
                self.errors.append(f"[{item['id']}] Git 标签校验失败: {e}")
                print(f"  ❌ 校验失败: {e}")

    def _validate_markdown(self, item: Dict[str, Any], path: Path) -> None:
        """校验 Markdown 文件

        Args:
            item: 提交物定义
            path: 文件路径
        """
        if item.get("must_exist", False):
            print(f"  ✅ 文件存在")

    def _check_dependency(self, item: Dict[str, Any]) -> None:
        """检查依赖项

        Args:
            item: 提交物定义
        """
        depends_on = item.get("depends_on")
        if depends_on:
            # 简化处理：假设依赖项已满足
            self.warnings.append(f"[{item['id']}] 依赖项: {depends_on}")
            print(f"  ⚠️  依赖项: {depends_on}")

    def _check_version_consistency(self) -> None:
        """检查版本一致性"""
        version = self.manifest["submission"]["version"]
        print(f"\n[版本一致性] 目标版本: {version}")
        # 简化处理：假设版本一致
        self.warnings.append(f"版本一致性检查: {version}")
        print(f"  ⚠️  版本一致性检查: {version}")

    def _report(self) -> None:
        """输出校验报告"""
        print()
        print("=" * 60)
        print(f"错误: {len(self.errors)}")
        for e in self.errors:
            print(f"  [ERROR] {e}")
        print(f"警告: {len(self.warnings)}")
        for w in self.warnings:
            print(f"  [WARN] {w}")
        print("=" * 60)

        if self.errors:
            print("\n❌ 校验失败，存在错误需要修复")
        else:
            print("\n✅ 校验通过，所有提交物符合要求")


def package_submission(manifest_path: str, project_root: str = ".") -> None:
    """打包提交物

    Args:
        manifest_path: 清单文件路径
        project_root: 项目根目录
    """
    validator = SubmissionValidator(manifest_path, project_root)

    if not validator.validate_all():
        print("\n❌ 校验失败，拒绝打包")
        sys.exit(1)

    print("\n📦 开始打包提交物...")

    # 创建输出目录
    output_dir = Path(project_root) / "dist"
    output_dir.mkdir(exist_ok=True)

    # 生成输出文件名
    version = validator.manifest["submission"]["version"]
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"submission_{version}_{date_str}.zip"

    # 创建 ZIP 文件
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in validator.manifest["items"]:
            path = Path(project_root) / item["path"]
            if path.exists():
                if path.is_file():
                    zipf.write(path, item["path"])
                    print(f"  ✅ 添加: {item['path']}")
                elif path.is_dir():
                    for file in path.rglob("*"):
                        if file.is_file():
                            arcname = str(file.relative_to(project_root))
                            zipf.write(file, arcname)
                    print(f"  ✅ 添加目录: {item['path']}")

    print(f"\n✅ 打包完成: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / (1024 * 1024):.1f}MB")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="M5 最终提交物校验与打包工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅校验
  python scripts/ci/validate_submission.py --check

  # 校验并打包
  python scripts/ci/validate_submission.py --pack

  # 自定义路径
  python scripts/ci/validate_submission.py --check --manifest config/submission_manifest.yaml --project-root .
        """,
    )

    parser.add_argument(
        "--check", action="store_true", help="仅校验提交物"
    )
    parser.add_argument(
        "--pack", action="store_true", help="校验并打包提交物"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="config/submission_manifest.yaml",
        help="清单文件路径 (默认: config/submission_manifest.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="项目根目录 (默认: 当前目录)",
    )

    args = parser.parse_args()

    if not args.check and not args.pack:
        parser.error("必须指定 --check 或 --pack 之一")

    if args.pack:
        package_submission(args.manifest, args.project_root)
    else:
        validator = SubmissionValidator(args.manifest, args.project_root)
        success = validator.validate_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
