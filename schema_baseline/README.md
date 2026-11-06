# 包 C：Schema 三元组请求适配版

`graphdpo_schema_baseline_v1_job` 是私有实验作业包。运行范围为 FB15k-237、WN18RR 的 10% 和 20%，TransE（200 维），种子 0/42/123，四次连续请求。只增加 Schema；不训练 P/R，不运行 MetaEU，不做修复探针。

## 已审核的算法定义

上游为 NKUShaw/KGUnlearningBySchema，固定 commit `a938d254ee76a2e475004ce90356ab0db9541eed`。

- 原更新循环：`upstream/Exp_FB15K237/Entity_TransE_FB15K237.py`。
- 原匹配器：`upstream/openke/data/EmbeddingDataLoader.py` 中 `CosineSchemaDataLoader.query_match_entity`。
- 首次请求从同种子 P 的实体和关系权重开始，后续继承上次 Schema 端点；每次请求新建空 Adam，lr=1e-5、weight_decay=0。
- 每次请求以当前起点实体表进行 KMeans（三簇、random_state=0），以当前保留事实构造邻接和候选；只对本次删除事实单遍更新。WN 使用同一 FB 示例参数，这是用户批准的参数迁移，不声称作者提供了 WN 默认配置。
- 每条原始正向删除事实至多尝试匹配 20 次；成功时最小化 `1-cos(e1-e2,e3-e4)` 并进行一次 Adam 更新。正逆方向都评价，但不把同一事实重复训练两次。
- 原示例中的 `train_times=1000` 与 margin loss 不进入这个自定义更新循环；不额外执行 GraphDPO 的 20 轮。损失只更新实体表，关系表逐次验证保持原样。
- 按关系建立候选索引只缩小搜索范围，保持同关系候选原顺序和 RNG 调用。保留原代码的 ID0 真值判断、邻接平行关系覆盖、第20次才匹配成功仍跳过等行为；报告 requested/updates/skipped，不把跳过等同于完成删除。
- 不读 R 排名或未来请求来决定当前更新。完整图与四次请求先做输入核对；每轮匹配只使用累计删除后的当前保留图。

审核原文保留在 `PLAN.zh.txt`，其中“尚未打包”和10项测试是上一轮审核时的历史状态；本包的当前测试以 `validation/TEST_RESULTS.txt` 为准。

## scikit-learn 与许可证

原仓库没有锁定 scikit-learn 版本。沿用已审核本地环境固定 **scikit-learn 1.6.1**，原调用 `KMeans(n_clusters=3, random_state=0)` 不变；在该版本中 `n_init='auto'`、`init='k-means++'`，实际初始化次数为1，算法为 lloyd。检查脚本核对这些解析值。旧版本可能默认10次初始化；本包不声称还原了作者未记录的旧环境。numpy 固定1.24.3、scipy1.12.0、joblib1.4.2、threadpoolctl3.6.0。实际服务器运行要求 PyTorch2.1.1+CUDA12.1；推荐 Python3.10。`setup.sh` 只创建本包的 `.venv` 并安装上述 CPU 依赖，不改旧环境。

Schema 未见仓库级许可证。本包只含本次私有实验需要的两个原文件，保留固定来源和 SHA256，**不可放入 `kge-unlearning-audit` 或作为 MIT 代码公开**。详见 `LICENSE_NOTICE.txt`。

## 数据、P/R 与包 A 的依赖

四个官方 ZIP 由启动参数传入，不在本包内：`fb15k-237-10.zip`、`wn18rr-10.zip`、`fb15k-237-20.zip`、`wn18rr-20.zip`。

| 配置 | 每次请求原始事实数 | Schema更新阶段 | Schema评价阶段 | 新P冻结评价 |
|---|---:|---:|---:|---:|
| FB10 | 31,011 | 12 | 12 | 0 |
| WN10 | 9,300 | 12 | 12 | 0 |
| FB20 | 62,023 | 12 | 12 | 12 |
| WN20 | 18,600 | 12 | 12 | 12 |

总计48次 Schema 单遍更新、48次 Schema 冻结评价、24次20% P冻结评价，另有四份 CPU 审计。20% 新评 P 是对旧权重按该比例查询集合重评，不重新训练。

`P_PINS.json` 与包 A v2 完全相同。10% R/P 使用 `REFERENCES10.json` 已有路径、模型绑定、commit/result/cache SHA256；历史缓存按查询身份对齐，不混淆保留和删除集合。主要来源为 FB 历史基线矩阵及 phase2 重训参照、WN 主实验及历史验证；每个种子/阶段的精确位置都在清单中。

20% 用参数 `--a20-session` 指向包 A v2 **已创建的具体 SESSION**，不是包目录。只读以下文件，不读它的进程、latest 指针、日志，不修改任何内容：

```text
<A_SESSION>/<FB或WN>/stages/train_R_<seed>_p<phase>/commit.json
<A_SESSION>/<FB或WN>/stages/train_R_<seed>_p<phase>/result.json
<A_SESSION>/<FB或WN>/stages/evaluate_R_<seed>_p<phase>/commit.json
<A_SESSION>/<FB或WN>/stages/evaluate_R_<seed>_p<phase>/result.json
<A_SESSION>/<FB或WN>/stages/evaluate_R_<seed>_p<phase>/evaluation.json
<A_SESSION>/<FB或WN>/stages/evaluate_R_<seed>_p<phase>/ranks.u32le
```

核对 R 训练检查点摘要与 R 评价输入摘要一致，但不加载 R 模型。缺少封口时每300秒检查，等待不持共享显存锁；存在封口但身份/哈希错误或 R 资源失败时停止，不把它当作一直未就绪。

**20% 的 Schema 训练和评价不依赖 R。** 默认一个 worker，依次 FB10→WN10→FB20→WN20；到达20%且GPU准入满足即可开始，无须等包 A 全部方法完成。所有 GPU 阶段之后，分别生成就绪配置的审计；未就绪配置等待自己的12份 R 评价。包 A 的其他遗忘方法不构成本包依赖。

## 评价、报告和事前预测

`igpu/audit.py`、排名和交叉拟合内核直接复制包 A v2，字节一致。完整过滤为主，旧过滤为附表；当前保留/累计删除 × 正向/逆向/combined，MR/MRR/Hit1/3/10、MRR_Avg、MRR_F1。独立 R_a 定档，R_b 比较，R_c 为 R_control，P_b 定重点格，六种有序角色等权；按删除来源阶段拆分，不筛负值。正 signed 表示比 R_b 检索更弱，绝对偏差先逐查询取绝对值再平均。

本包 Schema 额外评价原 GraphDPO preferred 的指定侧检索，区分全体/未观测；10% 复用的 P/R 未补做 preferred 推理，不生成虚构的该项对照值。空单元格保持 null；缺种子不冒充三种子均值。

运行前把 `PREDICTIONS.txt`、`protocol.json`、`PACKAGE_SHA256.json` 复制到新 SESSION，`admit.json` 记录三者 SHA256。预测不作为门槛，不自动增加实验。

输出在本包 `results/run_<时间>_<随机串>/`，每个配置独立目录：

```text
FB10/、WN10/、FB20/、WN20/
  stages/train_Schema_<seed>_p<phase>/final.pt、result.json、commit.json
  stages/evaluate_Schema_<seed>_p<phase>/evaluation.json、ranks.u32le、preferred.u32le
  audit/task_metrics_by_seed.csv、task_metrics_mean.csv
  audit/cross_bins_by_combination.csv、cross_bins_by_seed.csv、cross_bins_mean.csv
  audit/all_rank_metrics_by_seed.csv、preferred_by_seed.csv、preferred_mean.csv
  audit/coverage.csv、stage_resources.csv
```

表5/C2合并时以目录区分比例，表内 method 为 Schema，phase0–3 对应请求1–4。`report.sh` 显示四配置封口数、当前状态/阶段和心跳，生成 `results/run_.../report_...zip`，打印 ZIP SHA256。小 ZIP 只有统计 CSV、JSON、说明和预测，不含 `.pt` 或排名缓存。`COMPLETE_WITH_UNAVAILABLE` 表示流程结束但有资源限制/发散/缺方法结果；只有 `COMPLETE` 才表示所有配置覆盖完整。

## GPU 调度和错误边界

共享锁固定为 `/mnt/new4T/lsb/.graphdpo_gpu0_admission.lock`。flock 非阻塞轮询，锁内查显存并启动，启动后45秒释放。空闲显存不足即放锁、60秒后再查，600秒心跳；允许其他计算进程存在。首阶段门槛4096MiB，后续同类按实测峰值×1.5（沿用 A 的总显存上限处理）。不搜索、接管、停止 A/B/ZeroFisher/YAGO 或其他用户的进程。

每个阶段7200秒上限。确认基线自身 OOM 时直接 RESOURCE_LIMIT；观察到竞争时按初次+最多3次 OOM 重试，失败目录移到本包 resource_retries 保留，确定性输入/已完成更新前缀/已有排名前缀必须通过桥接。超限记 RESOURCE_LIMIT，后继依赖跳过；NaN/Inf 记 DIVERGED；数据/哈希/配对/流程错误立即停止并保留现场。超时只结束本包创建的那个子进程。

`launch.sh` 用 nohup+setsid，断开 SSH 不影响；服务器重启不保证自动恢复。当前包无自动改写/清理失败会话的命令，不要重复启动造成重复计算。只有本包自己的 coordinator 锁阻止重复启动，不以其他作业 PID 判重。

## 安装和运行

完整可复制命令见 `RUN_COMMANDS.txt`。使用前核对 ZIP SHA256，解压到新的同名目录；创建独立环境后执行 `bash check.sh`。check 禁用 CUDA，验证源文件和合成测试；传入四 ZIP 参数还会核对实际图数据。再把包 A 已打印的准确 SESSION 传给 launch。没有上传、启动或真实服务器检查在本地交付过程中发生。

CPU 本地测试环境与服务器 CUDA 环境不同，不能把本地通过当作真实服务器完成；见 `validation/TEST_RESULTS.txt`。预计约 **11–34 GPU worker小时**，包含 KMeans/匹配的 CPU 等待，排队、竞争、R依赖等待另计。完整图未实测 Schema，用于排程的估计见 `RESOURCE_ESTIMATE.txt`。
