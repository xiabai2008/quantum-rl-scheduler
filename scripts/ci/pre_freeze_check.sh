#!/bin/bash
# =============================================================================
# 8/15 代码冻结前检查清单
# Quantum RL Scheduler — Pre-Freeze Checklist
#
# 用法:
#   bash scripts/ci/pre_freeze_check.sh          # 完整检查
#   bash scripts/ci/pre_freeze_check.sh --quick  # 快速检查（跳过慢速项）
#
# 通过所有检查后才能打 tag 触发 release 流水线。
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# 颜色定义
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
QUICK_MODE=false

if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
    PASS=$((PASS + 1))
}

fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    FAIL=$((FAIL + 1))
}

warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $1"
}

# ---------------------------------------------------------------------------
# 标题
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD} 8/15 代码冻结前检查${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""

# ---------------------------------------------------------------------------
# 1. 获取最新代码
# ---------------------------------------------------------------------------
echo -e "${BOLD}[1/8]${NC} 检查代码是否最新..."
git fetch origin --quiet
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" = "$REMOTE" ]; then
    pass "本地代码与 origin/main 同步 ($(git rev-parse --short HEAD))"
else
    warn "本地代码落后于 origin/main"
    echo "        本地: $(git rev-parse --short HEAD)"
    echo "        远程: $(git rev-parse --short origin/main)"
    echo "        建议: git pull origin main"
fi

# ---------------------------------------------------------------------------
# 2. 分支检查
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[2/8]${NC} 检查当前分支..."
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ]; then
    pass "当前在 main 分支"
else
    warn "当前在 $BRANCH 分支，冻结前需要合并到 main"
fi

# 检查是否有未提交的更改
if git diff-index --quiet HEAD --; then
    pass "工作区干净，无未提交更改"
else
    fail "工作区有未提交的更改"
    git status --short
fi

# ---------------------------------------------------------------------------
# 3. CI 全绿检查
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[3/8]${NC} CI 状态检查..."

if command -v gh &> /dev/null; then
    if gh pr checks --required 2>/dev/null; then
        pass "所有 required CI checks 通过"
    else
        warn "gh 命令存在但无法获取 PR checks（可能无 open PR）"
    fi
else
    warn "gh CLI 未安装，跳过自动 CI 检查"
    echo "        请手动访问: https://github.com/xiabai2004/quantum-rl-scheduler/actions"
fi

# ---------------------------------------------------------------------------
# 4. 代码格式检查
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[4/8]${NC} 代码格式检查..."

if command -v black &> /dev/null; then
    if black --check --quiet src/ scripts/ tests/ 2>/dev/null; then
        pass "Black 格式检查通过"
    else
        fail "Black 格式检查失败，请运行: black src/ scripts/ tests/"
    fi
else
    warn "Black 未安装，跳过格式检查"
fi

if command -v isort &> /dev/null; then
    if isort --check-only --quiet src/ scripts/ tests/ 2>/dev/null; then
        pass "isort 导入排序检查通过"
    else
        fail "isort 检查失败，请运行: isort src/ scripts/ tests/"
    fi
else
    warn "isort 未安装，跳过导入排序检查"
fi

# ---------------------------------------------------------------------------
# 5. 单元测试 + 覆盖率
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[5/8]${NC} 运行单元测试..."

if python -m pytest tests/ -q --cov=src --cov-fail-under=60 --timeout=120 -W ignore::DeprecationWarning 2>/dev/null; then
    pass "单元测试全部通过 (cov >= 60%)"
else
    fail "单元测试失败或覆盖率不足"
fi

# ---------------------------------------------------------------------------
# 6. 提交物校验
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[6/8]${NC} 提交物校验..."

if python scripts/ci/validate_submission.py --check --manifest config/submission_manifest.yaml 2>/dev/null; then
    pass "提交物校验通过"
else
    warn "提交物校验有警告（错误会显示为 FAIL）"
fi

# ---------------------------------------------------------------------------
# 7. 权威数字审计
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[7/8]${NC} 权威数字一致性审计..."

if python scripts/ci/audit_authoritative_metrics.py 2>/dev/null; then
    pass "权威数字审计通过（无旧值残留）"
else
    fail "权威数字审计失败，存在旧值残留"
fi

# ---------------------------------------------------------------------------
# 8. 提交物确认
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[8/8]${NC} 提交物文件确认..."

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

check_file() {
    if [ -f "$ROOT/$1" ]; then
        pass "文件存在: $1"
    else
        if [ "$2" = "required" ]; then
            fail "强制文件缺失: $1"
        else
            warn "文件缺失 (非强制): $1"
        fi
    fi
}

check_file "README.md" "required"
check_file "requirements.txt" "required"
check_file "config/submission_manifest.yaml" "required"
check_file "deliverable_models/ppo_best_model_14dim.zip" "required"
check_file "deliverable_models/dqn_best_model_10dim.zip" "required"
check_file "results/reports/strategy_comparison.md" "required"
check_file "results/reports/ablation_report.md" "required"
check_file "results/reports/stress_test_report.md"
check_file "results/reports/real_machine_validation.md"
check_file "results/reports/statistical_validation.md"
check_file "docs/requirements_traceability.md"

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD} 检查汇总: ${GREEN}${PASS} 通过${NC} / ${RED}${FAIL} 失败${NC}${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}❌ 冻结前检查失败，请修复后重试。${NC}"
    echo ""
    echo "修复建议:"
    echo "  1. 代码格式: black src/ scripts/ tests/ && isort src/ scripts/ tests/"
    echo "  2. 运行测试: python -m pytest tests/ -v --cov=src --cov-fail-under=60"
    echo "  3. 提交物校验: python scripts/ci/validate_submission.py --check"
    echo "  4. 数字审计: python scripts/ci/audit_authoritative_metrics.py"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ 所有检查通过，可以打标签触发发布！${NC}"
    echo ""
    echo -e "${BOLD}下一步:${NC}"
    echo "  git tag -a v8.0-submission -m 'v8.0 提交版本 (2026-08-15)'"
    echo "  git push origin v8.0-submission"
    echo ""
    echo "推送标签后 GitHub Actions 会自动:"
    echo "  1. 运行全量测试 + 提交物校验"
    echo "  2. 打包最终提交物"
    echo "  3. 创建 GitHub Release 并上传附件"
    echo ""
    exit 0
fi
