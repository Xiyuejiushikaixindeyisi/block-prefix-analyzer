# block-prefix-analyzer

离线分析 LLM 请求 trace 中 **block 级前缀复用** 的工具。

当前状态：**V0.0.1 — 仅骨架**。核心算法尚未实现，API 随时可能变动。

## 设计目标
- 给定按时间排序的请求流，回放并输出：
  - 每个请求可复用的最长 **block 前缀**
  - 在"无限容量前缀缓存"假设下的 **理想命中率**（prefix-aware）
  - 在"只要出现过即可复用"假设下的 **block 级可复用率**
  - 可选 token 级前缀命中、reuse time、block lifespan 等衍生指标
- 与 vLLM / vLLM-Ascend 的语义逐步对齐，但核心分析器不依赖在线框架
- 离线、确定性、可单元测试、模块可替换

## 版本里程碑
- **V1（当前）**：只处理已包含 `block_hash_ids` 的输入；顺序回放；简单 Trie；基础报表
- **V2**：接入 tokenizer / chat template / block hash 复现，与框架对齐
- **V3**：性能与规模（radix 压缩、磁盘索引、并行预处理 等）

## 目录结构
```
src/block_prefix_analyzer/
  types.py          # 规范化数据模型 RequestRecord
  replay.py         # 时间序回放引擎（先算指标、再插入）
  metrics.py        # 指标定义与计算
  index/
    base.py         # PrefixIndex 抽象协议
    trie.py         # 简单 Trie 实现
  io/
    jsonl_loader.py # JSONL trace 读取
  reports/
    summary.py      # 汇总统计
tests/              # 与 src 模块一一对应的测试文件
```

## 开发环境
- Windows + WSL
- Python ≥ 3.10

## 快速开始
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## 下一步
查看 [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) 了解分步实施计划。
详细设计约束参见 [PROJECT_SPEC_FOR_CLAUDE.md](./PROJECT_SPEC_FOR_CLAUDE.md) 与 [CLAUDE.md](./CLAUDE.md)。
