# 代码讲解 07 · `summarize.py`：把 15 份成绩单变成一张表

> 位置：链路的最末端。所有 run 都跑完后，它负责读结果、算均值方差、出汇总表和四张对比图。
> 一句话职责：**扫描 `outputs/results/*`，按实验分组统计，生成 CSV / Markdown / 柱状图。**

---

## 一、这个文件在链路中的位置

```
outputs/results/<run_id>/
    config.json         每次跑的配置
    metrics.json        验证集成绩、参数量、训练时长
    test_metrics.json   测试集成绩
        │
        ▼
summarize.py  ← 你在这一篇
    load_runs()   扫描并读取 15 个目录
    summarize()   分组成 E1/E2/E6/E7，算均值±标准差，画图
        │
        ▼
outputs/summary/
    runs.csv                  每个 run 一行（原始数据）
    experiment_summary.csv    分组统计
    experiment_summary.md     给人看的表格
    fig_E1.png ... fig_E7.png 每组一张对比图
```

它**不训练、不评估**，只做“统计 + 展示”。删掉它，实验仍然成立，只是你得手动打开 15 个 json 去抄数字。

---

## 二、函数清单

| 函数 | 干什么 |
|---|---|
| `load_runs(out_root)` | 扫描 `results/` 下所有 run，读三个 json，拼成一行行字典 |
| `write_runs_csv(rows, path)` | 把每个 run 写进 `runs.csv` |
| `mean_std(values)` | 算均值和样本标准差（`ddof=1`） |
| `summarize(rows, out_dir)` | 按实验分组统计、写 CSV/Markdown、画对比图 |
| `main()` | 参数解析 + 调用上面几步 |

文件开头还有一个字典 `EXPERIMENTS`，定义“哪些 run 属于哪组实验”。

---

## 三、逐段讲关键代码

### 3.1 `EXPERIMENTS`：用前缀给 run 分组

```python
"E1": {
    "title": "E1 数据增强：开 vs 关",
    "groups": {
        "无增强": lambda rid: rid.startswith("cnn_gap_aug_off"),
        "有增强": lambda rid: rid.startswith("cnn_gap_aug_on"),
    },
},
```

- `rid` 是 run_id，比如 `cnn_gap_aug_on_s43`。这里用**字符串前缀**判断它属于哪一组。
- 这种写法的好处：不用维护一张“run_id → 组”的表，只要命名规范就能自动归类。
- 坏处：**命名一改，分组就失效**。比如你把 `cnn_gap_aug_on` 改成 `gap_aug_on`，E1 里“有增强”这组会变成 0 个 run，图也画不出来。
- 注意分组是“包含”关系：同一个 `cnn_gap_aug_on` 同时出现在 E1、E2、E7 里，这是正常的——它要分别和三个不同的对手比较。

### 3.2 `load_runs()`：只认“有成绩”的目录

```python
for run_dir in sorted((out_root / "results").iterdir()):
    if not run_dir.is_dir():
        continue
    cfg_p, met_p = run_dir / "config.json", run_dir / "metrics.json"
    if not cfg_p.exists() or not met_p.exists():
        continue
    ...
    test = json.loads(test_p.read_text(...)) if test_p.exists() else {}
```

- 目录里必须有 `config.json` 和 `metrics.json` 才认。只有 `best.pt`、没写 metrics 的“半成品”会被忽略。
- `test_metrics.json` 是可选的：没有就填 `nan`，后面统计时会被过滤掉。这就是为什么只训练不测试时表里会缺数。
- 每行拼出来的字段：run_id、模型、是否增强、阶段、种子、epoch 数、参数量、训练时长、验证准确率、测试准确率、测试 loss。
- **删掉 `sorted`**：不影响结果，只是 `runs.csv` 的行序每次不一样，不方便 diff。

### 3.3 `mean_std()`：为什么用 `ddof=1`

```python
arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
```

- 3 个种子得到 3 个测试准确率，报告里写“均值 ± 标准差”。
- `ddof=1` 是**样本标准差**（除以 n−1），不是总体标准差（除以 n）。样本量只有 3 时，两者差别不小；学术报告通常用样本标准差。
- 只有一个值时标准差记为 0，避免 `nan`。
- 先过滤 `nan`：某个 run 没做测试评估时不会把整组均值带成 nan。
- **如果不算方差只报单个最好成绩**：容易“挑好看的”，评审会认为不严谨。三种子均值±标准差是更诚实的写法。

### 3.4 `summarize()`：三件产物

#### ① 分组统计表

```python
subset = [r for r in rows if match(r["run_id"]) and not np.isnan(r["test_acc"])]
test_m, test_s = mean_std([r["test_acc"] for r in subset])
...
summary_rows.append({...})
```

- 对每个实验的每个变体，统计：种子数、参数量、测试准确率均值/标准差、验证准确率均值/标准差、训练时长均值/标准差、平均 epoch。
- 参数量取 `subset[0]["params_M"]`——同一组里参数都一样，取第一个即可。

#### ② `runs.csv` 与 `experiment_summary.csv`

```python
with path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
```

- `utf-8-sig` 带 BOM，**用 Excel 直接双击打开不会中文乱码**。这是给非程序员看的关键细节。
- `runs.csv` 是 15 行原始数据，`experiment_summary.csv` 是 8 行分组统计（E1 两组、E2 两组、E6 两组、E7 两组）。

#### ③ Markdown 表

```python
lines = ["| 实验 | 配置 | 种子数 | 参数量(M) | 测试准确率 | 验证准确率 | 训练时长(min) | 平均 epoch |", ...]
```

- 直接拼 Markdown 表格，方便贴进报告。百分比保留两位小数。
- **这是最终报告里那张表的来源**，数字对不上时先看这里。

#### ④ 四张对比图

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(labels, accs, yerr=errs, capsize=6, ...)
axes[0].set_ylim(max(0, min(accs) - 8), 100)
...
axes[1].bar(labels, secs, ...)
```

- 左图：测试准确率柱状图 + **误差棒**（`yerr=errs`，即标准差）。误差棒让“稳定性”可见——E1 里无增强那组误差棒特别长，一眼就能看出它不稳定。
- 右图：训练时长。用来回答“这个提升值不值这个时间”。
- `set_ylim(min-8, 100)`：如果从 0 开始画，90% 和 98% 看起来差不多；截断后差异更明显。**但截断坐标轴要谨慎**，报告里要说明，避免误导。
- 只有该实验至少有两个变体时才画（`if len(subset) < 2: continue`），否则一根柱子没什么可比。
- **删掉 `fig.savefig`**：图只在内存里，`summary/` 下什么都没有。

### 3.5 `main()`

```python
out_dir = out_root / "summary"
out_dir.mkdir(parents=True, exist_ok=True)
rows = load_runs(out_root)
if not rows:
    raise SystemExit("没有找到任何 run 结果")
```

- 先建目录、再扫描。
- 一个 run 都没有时直接退出并给出中文提示，而不是抛一个难懂的 KeyError。
- 注意 `write_runs_csv` 里用了 `rows[0].keys()`，所以必须在确认 `rows` 非空之后调用——顺序不能反。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 加一组新实验 | `EXPERIMENTS` 里加一个键（如 `"E3"`） | 同时要在 `run_experiments.py` 生成对应 run |
| 改分组规则 | `EXPERIMENTS` 里的 lambda | 按 run_id 前缀匹配 |
| 改统计指标（如加 F1） | `load_runs()` 读字段 + `summarize()` 计算 | 记得同时改 CSV 表头 |
| 改图的大小/配色 | `figsize`、`color=[...]` | 两根柱子一组，颜色数组长度为 2 |
| 改纵轴范围 | `set_ylim(max(0, min(accs) - 8), 100)` | 改成 0 起步更“诚实”，但差异不明显 |
| 改图的清晰度 | `savefig(..., dpi=150)` | 报告用 200 |
| 多输出一张“准确率 vs 参数量”散点图 | 在 `summarize()` 末尾加一段 | 数据都在 `summary_rows` 里 |
| 改中文表头 | Markdown 表头那一行 | CSV 表头是英文键名，改的话要同步 |
| 换输出目录 | `--out-root` | 默认是 `../outputs` |

---

## 五、新手易错点

1. **改了 run_id 命名，忘了改 `EXPERIMENTS`。**
   汇总表会少组、图为空。命名和分组是“约定”，两边必须一致。

2. **只训练没测试就汇总。**
   `test_acc` 是 `nan`，被过滤后那组可能只剩 0 个 run，直接从表里消失。先确认 `test_metrics.json` 都存在。

3. **`results/` 里混进了无关目录。**
   只要目录里有 `config.json` + `metrics.json` 就会被统计。手动拷贝的实验、smoke 覆盖的结果都会进来。跑汇总前最好确认 15 个目录都是正式结果。

4. **把最好种子的成绩写进报告。**
   报告应该写“均值 ± 标准差”，不是“我挑了个最高的”。三种子就是为了防止这种偏差。

5. **直接双击 CSV 看到乱码。**
   本项目用 `utf-8-sig` 写，Excel 能正常识别。如果你自己改成 `utf-8`，Excel 会把中文显示成乱码。
