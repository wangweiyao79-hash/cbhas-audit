# CB-HAS 论文回填对照表

生成时间：2026-07-07  
来源文件：FINAL_results.csv, FINAL_tvae_ce_perclass.csv, ABLATION_architecture.csv,
         table2_class_distribution.csv, aug_report_tvae.csv, env_info.txt, config.yaml,
         augment_tvae.py（D18核实）, train.py（D60核实）

---

## 第0步：执行修复补丁（13处文本替换，先于数据回填）

补丁目的：CB-HAS SMOTE兜底支路实际实现（随机对线性插值，无k参数）与正文描述（k近邻插值）不符，
按用户提供的13处str_replace逐字执行，正文改为正确描述。D12和D27占位符由补丁作废，无需填值。

| 替换# | 说明 |
|---|---|
| 1 | 摘要：SMOTE近邻插值兜底 → 类内随机对线性插值兜底 |
| 2 | 引言贡献1：同上 |
| 3 | 3.1数据层：同上 |
| 4 | 图1文字描述：下支路SMOTE框 → 下支路插值兜底框 |
| 5 | 3.2.2整节替换：标题+正文+公式变量（x_zi→x_j）+删除k'=min(k,nk-1)句 |
| 6 | 3.2.3分流公式：\text{SMOTE} → \text{插值兜底} |
| 7 | 3.2.3清单句：SMOTE支路 → 插值兜底支路 |
| 8 | 算法1第10-12行：SMOTE实现 → 随机对插值实现 |
| 9 | 表2注释：{不增强,TVAE,SMOTE} → {不增强,TVAE,插值} |
| 10 | 表3行：SMOTE近邻数k → 兜底插值方式（无k参数） |
| 11 | 4.3 D27句：替换为事实陈述（CTGAN与CB-HAS采用相同插值兜底） |
| 12 | 4.4发现二：修订措辞，强调CTGAN与CB-HAS唯一差异在生成器 |
| 13 | 5结论：SMOTE稳健插值 → 随机对线性插值 |

**补丁附注——基线SMOTE**：train.py（`--sampler smote`）使用标准sklearn
`SMOTE(k_neighbors=k, random_state=seed)`（k=min(5,min_count-1)），是真正的k近邻插值，
与CB-HAS兜底支路（随机对插值）不同。两者的差异已体现在表5结果（SMOTE+CE基线采用k近邻插值）。

---

## 散落数值回填（D01–D60）

| Token | 填入值 | 来源文件 | 字段/计算方式 |
|---|---|---|---|
| D01 | 83.91% | FINAL_results.csv | tvae+ce, macro_f1×100 |
| D02 | 99.76% | FINAL_results.csv | tvae+ce, accuracy×100 |
| D03 | 90.22% | FINAL_results.csv | tvae+ce, macro_precision×100 |
| D04 | 13.77 pp | FINAL_results.csv | (tvae+ce − none+ce) macro_f1×100 = 0.137657×100 |
| D05 | 20.53 pp | FINAL_results.csv | (tvae+ce − adasyn+ce) macro_f1×100 = 0.205290×100 |
| D06 | 14.50 pp | FINAL_results.csv | (tvae+ce − tvae+fl/sqrt_inv) macro_f1×100 = 0.144956×100 |
| D07 | 6.26 pp | FINAL_results.csv | (tvae+ce − tvae+fl/fixed) macro_f1×100 = 0.062555×100 |
| D08 | 83.11% | table2_class_distribution.csv | BENIGN占比 |
| D09 | 11 | table2_class_distribution.csv | Heartbleed样本数 |
| D10 | 0.001 | 计算 | 11/2,520,798×100=0.000436%，写作"不足0.001%" |
| D11 | 约19万比一 | 计算 | 2,095,057/11=190,459，写作"约19万比一" |
| D12 | 【已由补丁替换10作废】 | — | 表3该行改为"兜底插值方式"描述 |
| D13 | 2,000 | config.yaml | tau: 2000 |
| D14 | 5,000 | config.yaml | n_target: 5000 |
| D15 | 【删除标记】 | aug_report_tvae.csv | Δn_k=N_target-n_k与正文一致（Bot: 5000-1168=3832✓等），删除核实标记 |
| D16 | Bot、SSH-Patator、Web Attack–Brute Force、Web Attack–XSS | aug_report_tvae.csv | method=TVAE的类别（n_before分别为1168,1931,882,392，均≥50） |
| D17 | Heartbleed、Infiltration、Web Attack–SQL Injection | aug_report_tvae.csv | method=SMOTE_fallback的类别（n_before分别为7,22,13，均<50） |
| D18 | 【删除标记】 | augment_tvae.py | clip_to_real_range()在第209行被调用，裁剪已启用，删除核实标记 |
| D19 | 133,903 | model.py（已验证） | CNNBiLSTM参数量 |
| D20 | 2 | config.yaml | gamma: 2 |
| D21 | 3.9.13 | results/env_info.txt | Python版本 |
| D22 | 1.6.1 | results/env_info.txt | scikit-learn版本 |
| D23 | 1.37.3 | results/env_info.txt | sdv版本 |
| D24 | 0.12.4 | results/env_info.txt | imblearn版本 |
| D25 | 1,512,478 | 计算 | 2,520,798×0.6=1,512,478（aug后训练集=1,543,063，差值=30,585条合成样本✓） |
| D26 | 504,160 | 计算 | 2,520,798×0.2=504,160（与test集规模相同） |
| D27 | 【已由补丁替换11作废】 | augment_ctgan.py | 改为事实陈述：CTGAN对极稀有类采用相同插值兜底，保证对比公平 |
| D28 | 300 | config.yaml | ctgan_epochs: 300（TVAE使用同参数键） |
| D29 | 500 | config.yaml | ctgan_batch: 500（TVAE使用同参数键） |
| D30 | 98.43% | FINAL_results.csv | none+ce, accuracy×100 |
| D31 | 1.32 pp | FINAL_results.csv | (tvae+ce − none+ce) accuracy×100 = 0.013234×100 |
| D32 | 70.14% | FINAL_results.csv | none+ce, macro_f1×100 |
| D33 | 13.77 pp | 同D04 | 与D04相同值 |
| D34 | ADASYN | FINAL_results.csv | ADASYN macro_f1=63.38% > SMOTE macro_f1=61.38% |
| D35 | 4.12 pp | FINAL_results.csv | (tvae+ce − ctgan+ce) macro_f1×100 = 0.041235×100 |
| D36 | 17.66 pp | FINAL_results.csv | (ctgan+ce − ctgan+fl) macro_f1×100 = 0.176586×100 |
| D37 | 100.00% | FINAL_tvae_ce_perclass.csv | Infiltration f1_score×100 |
| D38 | 18.75% | FINAL_tvae_ce_perclass.csv | Web Attack · Sql Injection f1_score×100 |
| D39 | 10.14% | FINAL_tvae_ce_perclass.csv | Web Attack · XSS f1_score×100 (=0.101449) |
| D40 | 38.97% | FINAL_results.csv | none+ce, recall_Bot×100 |
| D41 | 43.08% | FINAL_tvae_ce_perclass.csv | Bot recall×100 |
| D42 | 83.91% | ABLATION_architecture.csv | cnn_bilstm macro_f1×100 |
| D43 | 8.07 pp | ABLATION_architecture.csv | (cnn_bilstm − cnn_only) macro_f1×100 = 0.080698×100 |
| D44 | 11.38 pp | ABLATION_architecture.csv | (cnn_bilstm − bilstm_only) macro_f1×100 = 0.113819×100 |
| D45 | 79.86% | ABLATION_architecture.csv | cnn_only macro_precision×100 |
| D46 | 90.22% | ABLATION_architecture.csv | cnn_bilstm macro_precision×100 |
| D47 | 72.53% | ABLATION_architecture.csv | bilstm_only macro_f1×100 |
| D48 | 99.76% | ABLATION_architecture.csv | cnn_bilstm accuracy×100 |
| D49 | 98.51% | ABLATION_architecture.csv | cnn_only accuracy×100 |
| D50 | 98.53% | ABLATION_architecture.csv | bilstm_only accuracy×100 |
| D51 | 11.38 pp | ABLATION_architecture.csv | max差=cnn_bilstm−bilstm_only macro_f1×100（=D44） |
| D52 | 5.38% | FINAL_tvae_ce_perclass.csv | Web Attack · XSS recall×100 |
| D53 | 10.14% | FINAL_tvae_ce_perclass.csv | Web Attack · XSS f1_score×100（=D39） |
| D54 | Web Attack–Brute Force | 模型推理（get_xss_confusion.py） | XSS 130个测试样本中117个(90.0%)被预测为WA-BF，7个(5.4%)正确，符合预期 |
| D55 | 10.71% | FINAL_tvae_ce_perclass.csv | Web Attack · Sql Injection precision×100（=0.107143） |
| D56 | 18.75% | FINAL_tvae_ce_perclass.csv | Web Attack · Sql Injection f1_score×100（=D38） |
| D57 | 21 | table2_class_distribution.csv | Web Attack · Sql Injection总样本数 |
| D58 | 4 | FINAL_tvae_ce_perclass.csv | Web Attack · Sql Injection support（测试集）=4（21×0.2=4.2≈4） |
| D59 | 43.08% | FINAL_tvae_ce_perclass.csv | Bot recall×100（=D41） |
| D60 | torch.manual_seed(42)、numpy.random.seed(42)、torch.cuda.manual_seed_all(42)、cudnn.deterministic=True、cudnn.benchmark=False | src/train.py | 第276-281行 |

---

## 整表回填

### 表2 — 类别分布与增强路径（按样本数降序重排）

| 类别 | 样本数 | 占比(%) | 增强路径 |
|---|---|---|---|
| BENIGN | 2,095,057 | 83.11 | 不增强 |
| DoS Hulk | 172,846 | 6.86 | 不增强 |
| DDoS | 128,014 | 5.08 | 不增强 |
| PortScan | 90,694 | 3.60 | 不增强 |
| DoS GoldenEye | 10,286 | 0.41 | 不增强 |
| FTP-Patator | 5,931 | 0.24 | 不增强 |
| DoS slowloris | 5,385 | 0.21 | 不增强 |
| DoS Slowhttptest | 5,228 | 0.21 | 不增强 |
| SSH-Patator | 3,219 | 0.13 | TVAE |
| Bot | 1,948 | 0.08 | TVAE |
| Web Attack–Brute Force | 1,470 | 0.06 | TVAE |
| Web Attack–XSS | 652 | 0.03 | TVAE |
| Infiltration | 36 | 0.00 | 插值 |
| Web Attack–SQL Injection | 21 | 0.00 | 插值 |
| Heartbleed | 11 | 0.00 | 插值 |

注：原稿行序有误（PortScan/DDoS位置互换，SSH-Patator在DoS slowloris/Slowhttptest之前），已按样本数降序重排。

### 表4 — 9配置整体性能（%）

| 配置 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| none+CE | 98.43 | 74.18 | 70.16 | 70.14 | 70.16 |
| ROS+CE | 97.15 | 59.37 | 87.47 | 65.55 | 87.47 |
| SMOTE+CE | 94.93 | 58.25 | 86.80 | 61.38 | 86.80 |
| ADASYN+CE | 96.27 | 57.08 | 89.58 | 63.38 | 89.58 |
| CTGAN+CE | 99.69 | 83.50 | 87.89 | 79.79 | 87.89 |
| CTGAN+FL | 94.02 | 57.63 | 92.56 | 62.13 | 92.56 |
| CB-HAS+CE（本文） | **99.76** | **90.22** | **87.15** | **83.91** | **87.15** |
| CB-HAS+FL(√inv) | 97.86 | 68.83 | 87.57 | 69.41 | 87.57 |
| CB-HAS+FL(α=0.25) | 99.61 | 86.70 | 83.87 | 77.65 | 83.87 |

### 表5 — 稀有攻击类别逐类性能（%）

F1 = 2PR/(P+R)，P=R=0时F1=0，均由未舍入原始值计算。

| 类别 | none+CE R | none+CE F1 | SMOTE+CE R | SMOTE+CE F1 | CB-HAS+CE R | CB-HAS+CE F1 |
|---|---|---|---|---|---|---|
| Bot | 38.97 | 56.09 | 100.00 | 8.69 | 43.08 | 59.05 |
| Web Attack–Brute Force | 85.37 | 58.51 | 60.20 | 22.09 | 93.54 | 79.71 |
| Web Attack–XSS | 0.00 | 0.00 | 66.15 | 12.61 | 5.38 | 10.14 |
| Infiltration | 0.00 | 0.00 | 85.71 | 5.19 | 100.00 | 100.00 |
| Web Attack–SQL Injection | 0.00 | 0.00 | 50.00 | 1.19 | 75.00 | 18.75 |
| Heartbleed | 50.00 | 66.67 | 50.00 | 66.67 | 100.00 | 100.00 |

### 表6 — 架构消融实验结果（%）

| 架构 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 1D-CNN-BiLSTM（完整） | 99.76 | 90.22 | 87.15 | 83.91 | 87.15 |
| CNN-only（去BiLSTM） | 98.51 | 79.86 | 83.29 | 75.84 | 83.29 |
| BiLSTM-only（去CNN） | 98.53 | 76.03 | 81.76 | 72.53 | 81.76 |

---

## 附录D 一致性校验结果

| 校验 | 结果 |
|---|---|
| ① 摘要D01–D07 ≡ 表4对应值/差值 | **全部通过**（Python计算验证） |
| ② 表4 Macro-Recall ≡ Balanced Accuracy（逐行） | **全部通过** |
| ③ 4.6节D42–D51 ≡ 表6；4.5节D37–D41、4.7节D52–D59 ≡ 表5/perclass | **全部通过** |
| ④ 图标注数值 ≡ 表格（论文MD无嵌入图数值，由绘图脚本保证） | 跳过（图在外部PDF） |
| ⑤ pp差值由原始未舍入值计算 | **全部通过**（原始精度6位小数后舍入） |

---

## 残留占位标记说明

**正文无残留数据占位标记。** 文首版本说明块第7行含`【D##】`和`【--】`两处，为格式举例元文本，随版本说明块在翻译前整体删除（Appendix E第5条）。

以下7处为Appendix E人工待办，不属数据回填范围：
- `【文献待补，见参考文献[15]】`（2.3节）
- `【图1 文字描述——供绘图使用，待绘制】`（3.1节）
- `【卷期页码待人工核对】` ×2（参考文献[11][12]）
- `【待补充：...】` ×3（参考文献[13][14][15]）

---

## 附：图例标签更新提醒（Appendix E 第1条）

fig_config_comparison 和 fig_perclass_f1 中的 "TVAE+CE" 标签需改为 "CB-HAS+CE"，
通过修改 src/plot_figures.py 的 label 后重渲染实现，数据不动。
