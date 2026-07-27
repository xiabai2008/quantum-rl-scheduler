#!/usr/bin/env python
"""实验可复现性校验脚本。

验证 multiseed 评估 JSON 中的 config_hash 与配置内容一致，
并检查引用的模型文件是否存在。
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _compute_config_hash(config: dict[str, Any]) -> str:
    """计算实验配置的确定性 SHA-256 哈希。

    按字母序序列化配置字典，确保哈希结果与键的顺序无关。

    Args:
        config: 实验配置字典

    Returns:
        32 位小写十六进制 SHA-256 哈希字符串
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate(path: Path) -> bool:
    """验证实验数据文件的可复现性。

    检查 config_hash 与配置内容是否匹配，以及引用的模型文件是否存在。

    Args:
        path: rewards_multiseed.json 文件路径

    Returns:
        验证通过返回 True，否则返回 False
    """
    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}")
        return False

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {})
    stored_hash = config.pop("config_hash", None)

    if stored_hash is None:
        print("[ERROR] 缺少 config_hash 字段")
        return False

    # 重新计算哈希
    computed = _compute_config_hash(config)

    if stored_hash != computed:
        print(f"[ERROR] config_hash 不匹配: stored={stored_hash} != computed={computed}")
        return False

    # 模型存在性检查
    missing_models: list[str] = []
    for key in ("ppo_model", "dqn_model"):
        model_path_str = config.get(key, "")
        if not model_path_str:
            continue
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            # 相对于项目根目录解析
            root = Path(__file__).resolve().parents[2]
            model_path = root / model_path
        if not model_path.exists():
            missing_models.append(str(model_path))
            print(f"[WARN] 模型文件缺失: {key}={model_path}")

    if missing_models:
        print(f"[WARN] 共 {len(missing_models)} 个模型文件缺失")

    print("[PASS] experiment config hash and reproducibility validation passed")
    return True


def main() -> int:
    """命令行入口。"""
    root = Path(__file__).resolve().parents[2]
    target = root / "results" / "multiseed_evaluation" / "rewards_multiseed.json"

    # 支持命令行传入自定义路径
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])

    return 0 if validate(target) else 1


if __name__ == "__main__":
    sys.exit(main())
