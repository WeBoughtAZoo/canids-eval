# 固定 FPR 重定阈值分析（诊断）

`oracle@k%` = 在 **test 负样本**上重选 τ 使 FPR≤k%（**用了 test 标签，不可部署**，只作上界）。
`AUC-ROC / AUC-PR_norm` 与阈值无关，是表征质量的干净度量。

## loso_set01

| 模式 | n(run) | MCC | Recall | FPR | AUC-ROC | AUC-PR_norm |
|---|---|---|---|---|---|---|
| as-is | 1 | 0.4595 | 0.703 | 8.77% | 0.8417 | 0.4961 |
| oracle@1% | 1 | 0.4606 | 0.332 | 1.00% | 0.8417 | 0.4961 |

| seed | as-is MCC | as-is FPR | oracle@1% MCC | oracle@1% FPR | AUC-ROC |
|---|---|---|---|---|---|
| 42 | 0.4595 | 8.77% | 0.4606 | 1.00% | 0.8417 |

## loso_set02

| 模式 | n(run) | MCC | Recall | FPR | AUC-ROC | AUC-PR_norm |
|---|---|---|---|---|---|---|
| as-is | 1 | 0.7083 | 0.616 | 0.57% | 0.8416 | 0.6867 |
| oracle@1% | 1 | 0.6779 | 0.633 | 1.00% | 0.8416 | 0.6867 |

| seed | as-is MCC | as-is FPR | oracle@1% MCC | oracle@1% FPR | AUC-ROC |
|---|---|---|---|---|---|
| 42 | 0.7083 | 0.57% | 0.6779 | 1.00% | 0.8416 |

## loso_set03

| 模式 | n(run) | MCC | Recall | FPR | AUC-ROC | AUC-PR_norm |
|---|---|---|---|---|---|---|
| as-is | 1 | 0.6495 | 0.954 | 8.42% | 0.9867 | 0.9565 |
| oracle@1% | 1 | 0.9028 | 0.934 | 1.00% | 0.9867 | 0.9565 |

| seed | as-is MCC | as-is FPR | oracle@1% MCC | oracle@1% FPR | AUC-ROC |
|---|---|---|---|---|---|
| 42 | 0.6495 | 8.42% | 0.9028 | 1.00% | 0.9867 |

## loso_set04

| 模式 | n(run) | MCC | Recall | FPR | AUC-ROC | AUC-PR_norm |
|---|---|---|---|---|---|---|
| as-is | 1 | 0.4356 | 0.801 | 17.50% | 0.8406 | 0.2061 |
| oracle@1% | 1 | 0.0000 | 0.000 | 0.00% | 0.8406 | 0.2061 |

| seed | as-is MCC | as-is FPR | oracle@1% MCC | oracle@1% FPR | AUC-ROC |
|---|---|---|---|---|---|
| 42 | 0.4356 | 17.50% | 0.0000 | 0.00% | 0.8406 |

