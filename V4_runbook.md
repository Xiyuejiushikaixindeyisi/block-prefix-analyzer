# V4 实验操作手册

本手册覆盖模块一（离线 reuse_distance 分析）、模块二（生产时序回放）、模块三（受控并发扫描）的完整操作步骤。

**当前阶段假设**：无法访问 `/metrics` 端点，所有分析均基于客户端可观测数据（延迟、token 数、成功率）。

---

## 0. 前置准备

### 0.1 拉取最新代码

```bash
cd block-prefix-analyzer
git pull
```

> **必须执行**：模块二分析脚本 `analyze_production_replay.py` 和模块三图表 `figure_3a` 均在最新提交中新增，旧版本没有这些功能。

### 0.2 安装依赖

```bash
# 分析脚本依赖（离线机器）
pip install matplotlib

# 模块二/三数据采集依赖（运行实验的机器，需能访问 vLLM API）
pip install aiohttp
```

### 0.3 确认 CSV 列位置

本手册所有命令默认 CSV 格式如下：

| 列索引 | 内容 |
|---|---|
| 0 | request_id |
| 1 | user_id |
| 2 | raw_prompt（原始请求文本）|
| 3 | timestamp（请求时间戳）|

如果你的 CSV 列顺序不同，在对应命令中修改 `--col-*` 参数。

---

## 模块一：离线 KV Block reuse_distance 分析

> **适用场景**：只需要 JSONL 请求数据，无需 API，在任意机器上离线运行。

### 步骤 1：确认输入文件

```bash
ls data/internal/qwen_v3_5_27b_64k/requests.jsonl
```

### 步骤 2：运行分析

```bash
python scripts/generate_reuse_distance.py \
    configs/maas/qwen_v3_5_27b_64k/reuse_distance.yaml
```

**当前输出**（`available_cache_blocks` 未设置，无 LRU 阈值线）：

```
outputs/maas/qwen_v3_5_27b_64k/reuse_distance/
├── reuse_distance_events.csv     ← 每个复用事件的详细数据
├── reuse_distance_sorted.png     ← 图 1a：降序分布图
├── reuse_distance_cdf.png        ← 图 1b：CDF 图
└── metadata.json                 ← 汇总统计
```

### 步骤 3（事后补充）：填入 available_cache_blocks

当你从 vLLM 启动日志获取到 `num_gpu_blocks` 后：

```yaml
# 编辑 configs/maas/qwen_v3_5_27b_64k/reuse_distance.yaml
available_cache_blocks: <填入 num_gpu_blocks 的值>
```

重新运行步骤 2，图中将新增红色阈值线并输出 `evicted_fraction`（LRU 驱逐比例）。

---

## 模块二：生产时序回放

> **适用场景**：在能访问 vLLM API 的机器上运行。按生产数据的真实时间戳顺序发送请求，测量实际延迟和成功率随时间的变化。

### 数据流

```
原始 CSV
  → replay_production_benchmark.py（发送请求，收集客户端数据）
  → request_results.csv + metric_timeseries.csv
  → analyze_production_replay.py（生成图表）
  → Figure 2a（延迟时序）+ Figure 2b（吞吐/成功率）
```

### 步骤 1：运行时序回放

在**能访问 vLLM API 的机器**上执行：

```bash
python scripts/replay_production_benchmark.py \
    --input      data/internal/qwen_v3_5_27b_64k/raw/<你的文件名>.csv \
    --output     data/benchmark/m2_qwen_v3_5_27b_64k \
    --endpoint   http://<YOUR_ENDPOINT>:<PORT> \
    --model      <YOUR_MODEL_NAME> \
    --col-request-id 0 \
    --col-user-id    1 \
    --col-raw-prompt 2 \
    --col-timestamp  3 \
    --n          200 \
    --max-tokens 128 \
    --time-scale 0 \
    --has-header \
    --encoding   utf-8-sig
```

**参数说明**：

| 参数 | 含义 | 是否需要修改 |
|---|---|---|
| `--input` | 生产数据 CSV 文件路径 | **必须修改** |
| `--endpoint` | vLLM API 地址，含端口 | **必须修改** |
| `--model` | vLLM 注册的模型名，与 `/v1/models` 返回值一致 | **必须修改** |
| `--n 200` | 取前 200 行 | 可调整 |
| `--max-tokens 128` | 每条请求最多生成 128 token | 可调整 |
| `--time-scale 0` | 不按真实时间间隔等待（加速回放）| 建议保持 0 |
| `--has-header` | CSV 有表头行 | 视文件格式决定 |
| `--encoding utf-8-sig` | 处理 BOM 头编码 | 视文件格式决定 |

**运行中输出示例**（无 `/metrics` 时）：

```
batch   1/ 47  ts=2024-01-15 10:00:01  size= 4  (no /metrics)
batch   2/ 47  ts=2024-01-15 10:00:02  size= 6  (no /metrics)
...
```

**完成后输出**：

```
data/benchmark/m2_qwen_v3_5_27b_64k/
├── config.json               ← 本次实验参数记录
├── request_results.csv       ← 每条请求的延迟、token 数、成功/失败
└── metric_timeseries.csv     ← 每个并发批次的汇总统计
```

### 步骤 2：分析结果

可在**任意机器**上执行（不需要 API 访问）：

```bash
python scripts/analyze_production_replay.py \
    --input  data/benchmark/m2_qwen_v3_5_27b_64k \
    --output data/benchmark/m2_qwen_v3_5_27b_64k/analysis
```

**输出文件**：

```
data/benchmark/m2_qwen_v3_5_27b_64k/analysis/
├── summary.json                      ← 汇总统计（JSON）
├── figure_2a_latency_timeline.png    ← 每批次延迟随时间变化
└── figure_2b_throughput.png          ← 每批次吞吐率和成功率
```

> 如果后续开启 `/metrics`，重新运行步骤 1 后，分析时将自动额外生成 `figure_2c_hit_rate.png`。

---

## 模块三：受控并发扫描

> **适用场景**：固定 100 条 prompt，人工控制并发数（1/2/4/8/16），找到延迟开始上升的拐点。

### 数据流

```
原始 CSV
  → sample_prompts_for_benchmark.py（按长度分层采样）
  → sampled_prompts.jsonl
  → run_kv_cache_benchmark.py（并发扫描，收集客户端数据）
  → run_results.csv
  → analyze_benchmark_results.py（生成图表）
  → Figure 3a（延迟 × 吞吐双轴）
```

### 步骤 1：采样代表性 Prompt

```bash
python scripts/sample_prompts_for_benchmark.py \
    --input      data/internal/qwen_v3_5_27b_64k/raw/<你的文件名>.csv \
    --output     data/benchmark/sampled_prompts_qwen64k.jsonl \
    --col-request-id 0 \
    --col-user-id    1 \
    --col-raw-prompt 2 \
    --col-timestamp  3 \
    --n 100 \
    --seed 42 \
    --has-header \
    --encoding utf-8-sig
```

输出示例：
```
Sampled 100 prompts:
  p0-p25:   25 prompts  len=[120, 850]    median=412
  p25-p50:  25 prompts  len=[851, 2100]   median=1350
  p50-p75:  25 prompts  len=[2101, 6500]  median=4200
  p75-p100: 25 prompts  len=[6501, 42000] median=12000
```

### 步骤 2：并发扫描

在**能访问 vLLM API 的机器**上执行：

```bash
python scripts/run_kv_cache_benchmark.py \
    --prompts     data/benchmark/sampled_prompts_qwen64k.jsonl \
    --output      data/benchmark/m3_qwen_v3_5_27b_64k \
    --endpoint    http://<YOUR_ENDPOINT>:<PORT> \
    --model       <YOUR_MODEL_NAME> \
    --concurrency 1 2 4 8 16 \
    --rounds      3 \
    --max-tokens  64 \
    --timeout     180
```

**参数说明**：

| 参数 | 含义 | 是否需要修改 |
|---|---|---|
| `--endpoint` | vLLM API 地址 | **必须修改** |
| `--model` | 模型名称 | **必须修改** |
| `--concurrency 1 2 4 8 16` | 并发级别从低到高 | 可扩展（如加 32） |
| `--rounds 3` | 每个并发级别重复 3 次取均值 | 建议保持 3 |
| `--max-tokens 64` | 输出 token 数，保持短以控制变量 | 建议保持 64 |

**运行时长估算**（100 prompts × 5 并发级别 × 3 rounds）：
- 若平均延迟 5s：约 100 × 5 × 3 / min(并发, 100) ≈ 25–75 分钟
- 建议在开始前确认 API 可访问且不会超时

**运行中输出示例**（无 `/metrics` 时）：

```
concurrency=1  round=1/3 ...
  hit_rate=n/a  removed/stored=n/a  idle_mean=n/a  rps=0.21

concurrency=2  round=1/3 ...
  hit_rate=n/a  removed/stored=n/a  idle_mean=n/a  rps=0.38
```

**完成后输出**：

```
data/benchmark/m3_qwen_v3_5_27b_64k/
├── config.json       ← 实验配置
└── run_results.csv   ← 每个并发级别 × 每轮的详细数据
```

### 步骤 3：分析结果

可在**任意机器**上执行：

```bash
python scripts/analyze_benchmark_results.py \
    --results data/benchmark/m3_qwen_v3_5_27b_64k/run_results.csv \
    --output  data/benchmark/m3_qwen_v3_5_27b_64k/analysis
```

**输出文件**：

```
data/benchmark/m3_qwen_v3_5_27b_64k/analysis/
├── summary.json                          ← 拐点分析（JSON）
├── figure_3a_latency_throughput.png      ← 延迟 p50/p95 + 吞吐 vs 并发（主图）
└── capacity_boundary.png                 ← 4 合 1 图（无 /metrics 时指标为空）
```

> `figure_3a_latency_throughput.png` 是无需 `/metrics` 的主要结论图，横轴为并发数，左纵轴为延迟（p50/p95），右纵轴为吞吐（requests/s）。

---

## 实验顺序建议

```
阶段 A（离线，任意机器，已完成）
  └─ 模块一：generate_reuse_distance.py

阶段 B（需要访问 vLLM API 的机器）
  └─ 模块二步骤 1：replay_production_benchmark.py
  └─ 模块三步骤 1：sample_prompts_for_benchmark.py
  └─ 模块三步骤 2：run_kv_cache_benchmark.py

阶段 C（任意机器，分析输出数据）
  └─ 模块二步骤 2：analyze_production_replay.py
  └─ 模块三步骤 3：analyze_benchmark_results.py
```

---

## 实验完成后请提供的数据

### 模块二结果

请提供以下文件内容（或截图）：
- `data/benchmark/m2_qwen_v3_5_27b_64k/analysis/summary.json`
- `figure_2a_latency_timeline.png`
- `figure_2b_throughput.png`

### 模块三结果

请提供：
- `data/benchmark/m3_qwen_v3_5_27b_64k/run_results.csv`（或其 avg 行的截取）
- `figure_3a_latency_throughput.png`

### 后续补充参数（模块一精确化用）

当你能访问部署 vLLM 的服务器时，运行以下命令获取 KV cache 容量参数：

```bash
# 方法 1：从启动日志获取（推荐）
grep -E "num_gpu_blocks" /path/to/vllm_startup.log | tail -3

# 方法 2：从 /metrics 端点（若可访问）
curl -s http://<YOUR_ENDPOINT>:<PORT>/metrics | grep "gpu_cache_usage"

# 方法 3：记录 vLLM 启动参数
# 记下以下值：
#   --block-size（默认 16，Ascend prefix caching 可能为 128）
#   --gpu-memory-utilization
#   --max-model-len
```

将获取到的 `num_gpu_blocks` 值告知，即可完成模块一的 LRU 驱逐比例计算。
