"""一键复现脚本：不需要命令行参数，直接运行（编辑器 ▶ 或双击）即可跑完整流程。

流程：
  1. 数据检查      check_data.py
  2. 缓存生成      prepare_cache.py（缓存已存在则跳过）
  3. 迁移学习      run_experiments.py --only transfer          （6 个 run）
  4. 自搭 CNN E1   run_experiments.py --only e1                （6 个 run）
  5. 自搭 CNN E7   run_experiments.py --only e7                （3 个 run）
  6. 汇总图表      summarize.py

参数与 README 一致：
  - 随机种子 42 / 43 / 44
  - 迁移学习 batch 32（阶段 A 6 epoch、阶段 B 8 epoch）
  - E1 / E7 的自搭 CNN batch 48、12 epoch
  - 对需要重跑的 seed 加 --force，覆盖该 seed 的旧结果

续跑：脚本按 seed 检查 outputs/results/<run_id>/ 下的 metrics.json 与
test_metrics.json；已完整的 seed 直接跳过，所以中途按 Ctrl+C 或关掉窗口后，
重新运行本文件即可从断点继续（最多重跑当前那个 seed）。
若想彻底从零重跑，先手动删除 outputs/results 下对应目录。

产物位置（与原有脚本一致，本文件不改变任何已有行为）：
  outputs/results/<run_id>/   每个 run 的权重、曲线、混淆矩阵、错误样本
  outputs/summary/            汇总表与对比图
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------- 路径与参数（与 README 一致） ----------------
PROJECT_DIR = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_DIR.parent / "outputs"
RESULTS_DIR = OUT_ROOT / "results"
CACHE_DIR = PROJECT_DIR / "cache"
DATA_ROOT = Path(r"d:\DeepStudy\Project1\Cat_Dog_data")
TORCH_ENV_PYTHON = Path(r"D:\DeepStudy\Anaconda\envs\torch_env\python.exe")

SEEDS = [42, 43, 44]
TRANSFER_BATCH = 32
CNN_BATCH = 48
CNN_EPOCHS = 12
HEAVY_EPOCHS = 12

# 每个阶段期望产出的 run（命名必须与 run_experiments.py 一致）
TRANSFER_RUNS = {s: [f"resnet_stageA_s{s}", f"resnet_stageB_s{s}"] for s in SEEDS}
E1_RUNS = {s: [f"cnn_gap_aug_on_s{s}", f"cnn_gap_aug_off_s{s}"] for s in SEEDS}
E7_RUNS = {s: [f"cnn_heavy_aug_on_s{s}"] for s in SEEDS}


class StepError(RuntimeError):
    """某一步执行失败，用于统一提示“是哪一步出错”。"""


# ---------------- 解释器选择 ----------------
def _has_torch(python: str) -> bool:
    """只检查能否找到 torch 包，不真正 import（import 很慢）。"""
    code = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    try:
        return subprocess.run([python, "-c", code], capture_output=True, timeout=90).returncode == 0
    except Exception:
        return False


def pick_python() -> Path:
    """优先用当前解释器；双击运行时若它不是 torch_env，再尝试 README 记录的路径。"""
    candidates = [sys.executable, str(TORCH_ENV_PYTHON), shutil.which("python")]
    for p in candidates:
        if p and Path(p).exists() and _has_torch(str(p)):
            return Path(p)
    raise StepError(
        "找不到装好 PyTorch 的 Python 解释器。\n"
        f"请在编辑器中把解释器选为 {TORCH_ENV_PYTHON} 后再运行；\n"
        "或直接双击本文件前，确认该路径存在。"
    )


# ---------------- 子进程执行 ----------------
def run_command(cmd: list, title: str, estimate: str) -> str:
    """执行一条子命令，实时转发输出，失败时抛出 StepError。返回全部输出文本。"""
    print(f"\n{'-' * 70}")
    print(f"{title}（预计 {estimate}）")
    print(f"{'-' * 70}", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"   # 子进程统一用 UTF-8，避免中文乱码
    env["PYTHONUNBUFFERED"] = "1"       # 不缓冲，进度能实时显示

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print("  " + line, flush=True)
            lines.append(line)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    minutes = (time.perf_counter() - t0) / 60
    if code != 0:
        raise StepError(f"步骤「{title}」失败（退出码 {code}），请把上面的报错信息发给我。")
    print(f"  >> {title}完成，用时 {minutes:.1f} 分钟")
    return "\n".join(lines)


# ---------------- 结果完成度判断（续跑） ----------------
def _run_status(run_id: str) -> tuple[bool, bool]:
    """返回 (是否训练完成, 是否测试集评估完成)。"""
    d = RESULTS_DIR / run_id
    return (d / "metrics.json").exists(), (d / "test_metrics.json").exists()


def incomplete_seeds(group: dict[int, list[str]]) -> list[int]:
    """列出还没完整跑完（训练或测试评估缺任一）的 seed。"""
    seeds = []
    for seed, run_ids in group.items():
        if any(not all(_run_status(r)) for r in run_ids):
            seeds.append(seed)
    return seeds


def need_retrain(group: dict[int, list[str]], seeds: list[int]) -> bool:
    """要跑的 seed 里有缺 metrics.json 的 run → 需要加 --force 覆盖旧结果重训。"""
    return any(not _run_status(r)[0] for s in seeds for r in group[s])


def describe_group(group: dict[int, list[str]]) -> str:
    done = sum(1 for runs in group.values() for r in runs if all(_run_status(r)))
    total = sum(len(runs) for runs in group.values())
    return f"已完成 {done}/{total} 个 run"


# ---------------- 各阶段 ----------------
def step_check_data(python: Path) -> None:
    title = "[1/6] 数据检查"
    if not DATA_ROOT.is_dir():
        raise StepError(f"{title}失败：数据目录不存在：{DATA_ROOT}\n"
                        "请确认 Cat_Dog_data 放在 d:\\DeepStudy\\Project1 下。")
    out = run_command([python, "check_data.py", str(DATA_ROOT)], title,
                      "约 3–5 分钟（逐张校验 25000 张图）")
    if "检查通过" not in out:
        raise StepError(f"{title}未通过：上面的问题列表里有多条异常，数据可能不完整。\n"
                        f"请检查 {DATA_ROOT} 下是否为 train/cat、train/dog、test/cat、test/dog 四个目录。")


def step_prepare_cache(python: Path) -> None:
    title = "[2/6] 生成 256px 缓存"
    need = [CACHE_DIR / "train_256.npy", CACHE_DIR / "test_256.npy"]
    missing = [p for p in need if not p.exists()]
    if not missing:
        size_gb = sum(p.stat().st_size for p in need) / 1024 ** 3
        print(f"\n{'-' * 70}")
        print(f"{title}：缓存已存在，跳过（共 {size_gb:.2f} GB）")
        print(f"  位置：{CACHE_DIR}")
        return
    run_command([python, "prepare_cache.py"], title,
                "约 6 分钟（首次解码 25000 张图）")


def step_group(python: Path, title: str, group: dict[int, list[str]],
               extra_args: list[str], estimate: str) -> None:
    """跑一组 run：只跑未完成的 seed；需要重训时加 --force 覆盖旧结果。"""
    print(f"\n{'-' * 70}")
    print(f"{title}：{describe_group(group)}")
    seeds = incomplete_seeds(group)
    if not seeds:
        print("  全部完成，跳过。")
        return

    cmd = [python, "run_experiments.py", "--only", *extra_args[0], "--seeds",
           *[str(s) for s in seeds], "--batch-size", str(extra_args[1])]
    cmd += extra_args[2]
    if need_retrain(group, seeds):
        cmd.append("--force")  # 覆盖该 seed 的旧结果
        action = "重训（覆盖旧结果）"
    else:
        action = "只补测试集评估"
    print(f"  待处理 seed：{seeds}，动作：{action}")
    run_command(cmd, title, estimate)


def step_summarize(python: Path) -> None:
    run_command([python, "summarize.py"], "[6/6] 汇总结果与图表", "约 10 秒")


# ---------------- 主流程 ----------------
def main() -> None:
    print("=" * 70)
    print("猫狗图像分类 · 一键复现")
    print("=" * 70)
    print(f"项目目录：{PROJECT_DIR}")
    print(f"数据目录：{DATA_ROOT}")
    print(f"输出目录：{OUT_ROOT}")
    print(f"随机种子：{SEEDS}")
    print(f"迁移学习：batch {TRANSFER_BATCH}；自搭 CNN：batch {CNN_BATCH}，E1 {CNN_EPOCHS} epoch，"
          f"E7 {HEAVY_EPOCHS} epoch")
    print("提示：首次完整跑约 1.8–2 小时；中途中断后重新运行本文件即可续跑。")
    print("      （GPU 风扇转起来、控制台不断刷进度属于正常现象）")

    total_t0 = time.perf_counter()
    python = pick_python()
    print(f"\n使用的 Python：{python}")

    step_check_data(python)
    step_prepare_cache(python)

    step_group(python, "[3/6] 迁移学习 ResNet18（阶段 A + B）", TRANSFER_RUNS,
               [["transfer"], TRANSFER_BATCH, ["--epochs-a", "6", "--epochs-b", "8"]],
               "约 13 分钟（6 个 run）")
    step_group(python, "[4/6] 自搭 CNN · E1 数据增强对比", E1_RUNS,
               [["e1"], CNN_BATCH, ["--epochs-cnn", str(CNN_EPOCHS)]],
               "约 65 分钟（6 个 run）")
    step_group(python, "[5/6] 自搭 CNN · E7 轻量化对比", E7_RUNS,
               [["e7"], CNN_BATCH, ["--epochs-heavy", str(HEAVY_EPOCHS)]],
               "约 25 分钟（3 个 run）")

    step_summarize(python)

    total_min = (time.perf_counter() - total_t0) / 60
    print("\n" + "=" * 70)
    print(f"全部完成，总用时 {total_min:.1f} 分钟。")
    print("=" * 70)
    print(f"每个 run 的权重/曲线/混淆矩阵/错误样本：{RESULTS_DIR}")
    print(f"汇总表：{OUT_ROOT / 'summary' / 'experiment_summary.md'}")
    print(f"对比图：{OUT_ROOT / 'summary'} 下的 fig_E1.png / fig_E2.png / fig_E6.png / fig_E7.png")
    print(f"已有完整报告：{OUT_ROOT / 'PROJECT_REPORT.md'}")
    print("\n若想彻底从零重跑，删除 outputs/results 下对应 run 目录后再运行本文件。")


def _pause_before_exit() -> None:
    """双击运行时留住窗口，方便看结果；编辑器里直接回车即可。"""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input("\n按回车键关闭窗口...")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已手动中断。重新运行本文件即可从断点继续（已完成的 run 不会重跑）。")
        exit_code = 130
    except StepError as e:
        print(f"\n\n[中止] {e}")
        exit_code = 1
    except Exception as e:  # 兜底，避免双击后窗口一闪而过看不到错误
        import traceback
        traceback.print_exc()
        print(f"\n\n[中止] 未预期的错误：{e}")
        exit_code = 1
    finally:
        _pause_before_exit()

    sys.exit(exit_code)
