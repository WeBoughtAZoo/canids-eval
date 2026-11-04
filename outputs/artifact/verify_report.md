# 充分统计量自检报告

产物：`outputs/artifact`　批次：`main, gpu1`

**总计 1911 项，失败 0 项**

| 检查 | 比对项 | 失败 |
|---|---|---|
| C1 runs_index | 705 | 0 |
| C2 table_main | 60 | 0 |
| C3 table_density | 60 | 0 |
| C4 comparison | 240 | 0 |
| C5 metrics_json | 846 | 0 |

**全部通过**：计数表可逐位还原各批次的 `runs_index.csv` 与 `metrics_test.json`；
主批次的 `table_main.csv`、`table_density.csv`、`comparison*.csv` 亦逐格一致。