# StreamingVLA 运行结果与差距分析

> 本文只覆盖「结果统计 + 差距归因 + 解决方案」三件事，准备工作与问题修复清单见 `docs/summary1.md`。
> 重点在第四节的解决方案，每个方案都给出：验证目标、具体操作、判定标准、成本、风险。

- 评测日期：2026-10-26 ~ 10-28
- 代码：`/home/go2/StreamingVLA/StreamingVLA`（HEAD `644d356` + 本项目的修补）
- 硬件：单卡 RTX 4090D（24 GB），服务端与客户端分离进程
- 参照：论文 StreamingVLA 在 LIBERO 的 reported 平均成功率 **94.9%**，`T_action` **31.6 ms**

---

## 0. 一句话结论

**评测链路已验证可信**（π0.5 在同一套环境、同一批初始状态上拿到 **99%**），但 **StreamingVLA 实测只有 42%（AEO 开）/ 53%（AEO 关）**，与论文 94.9% 相差约 50 个点。经系统排查，`use_sfp`、归一化映射、`action_states` 语义、AEO 时序、初始状态难度均已排除；**当前首要怀疑对象是 HF 上发布的检查点本身（形似实验性运行，非论文最终权重）**，其次是归一化统计量数值与原训练数据不一致。第四节的方案 A（动作数值比对）用 10 分钟即可把这两者区分开。

---

## 1. 评测设置

| 项目 | 设置 |
| --- | --- |
| 套件 | `libero_spatial`（10 任务） |
| 每任务 trial 数 | **n=1**（冒烟）与 **n=10**（对照） |
| 初始状态 | `task_suite.get_task_init_states(i)`，即每任务的 `initial_states[0..n-1]` |
| seed | 7 |
| `num_steps_wait` | 10（等物体稳定，执行 dummy action） |
| 图像 | agentview + eye-in-hand（翻转 180°），`resize_with_pad` 到 224×224 |
| 步数上限 | `libero_spatial` = 220 步 |
| StreamingVLA | `replan_steps=10`、`action_left=2`、`threshold=4.43`、`use_judger=True`（论文配置）；另做 `use_judger=False` 消融 |
| π0.5 基线 | `replan_steps=10`（与 SVLA 的 action_horizon=10 对齐） |
| 服务端 | StreamingVLA → PyTorch 分支（含 AEO 预测器）；π0.5 → JAX 分支（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`） |
| 关键约束 | 两个服务端不可同时运行；均使用 `.venv/bin/python`（不用 `uv run`，避免触发 sync） |

---

## 2. 结果统计

### 2.1 总览（n=10/任务，共 100 episode）

| 配置 | 成功数 | 成功率 | 平均 episode 时长（仅成功） | 平均动作数（仅成功） | Action Speed |
| --- | --- | --- | --- | --- | --- |
| **π0.5 基线** | **99/100** | **99.0%** | 3.35 s | 103.56 | 30.93 a/s |
| StreamingVLA（AEO 开，论文配置） | 42/100 | **42.0%** | 3.45 s | 114.50 | 33.15 a/s |
| StreamingVLA（AEO 关） | 53/100 | **53.0%** | 3.51 s | 107.25 | 30.57 a/s |

> π0.5 论文值约 96–97%，本配置 99% 属于正常偏高的波动范围，说明仿真与评测协议无系统性偏差。

### 2.2 逐任务成功率（n=10）

| # | 任务（`pick up the black bowl … and place it on the plate`） | SVLA（AEO 开） | SVLA（AEO 关） | π0.5 |
| --- | --- | --- | --- | --- |
| 1 | between the plate and the ramekin | 9/10 | 9/10 | 10/10 |
| 2 | next to the ramekin | 1/10 | 1/10 | 10/10 |
| 3 | from table center | 5/10 | 8/10 | 10/10 |
| 4 | **on the cookie box** | **0/10** | 1/10 | 10/10 |
| 5 | in the top drawer of the wooden cabinet | 1/10 | 5/10 | 10/10 |
| 6 | on the ramekin | 10/10 | 9/10 | 9/10 |
| 7 | next to the cookie box | 1/10 | 3/10 | 10/10 |
| 8 | on the stove | 10/10 | 10/10 | 10/10 |
| 9 | next to the plate | 5/10 | 6/10 | 10/10 |
| 10 | **on the wooden cabinet** | **0/10** | 1/10 | 10/10 |

**分布特征**（这是最有信息量的部分）：

- 有 2 个任务是 **10/10 满分**（`on the ramekin`、`on the stove`），与 π0.5 持平 → 模型没有坏掉，基本抓取-放置能力在。
- 有 2 个任务是 **0–1/10**（`on the cookie box`、`on the wooden cabinet`）→ 这两个任务都要求把碗**举高放到台面/柜面上**，属于对末端执行器高度与末端姿态精度要求最高的一类。
- 其余任务在 1/10 ~ 9/10 之间散布。

这种「简单任务满分、需要精细高度的任务全灭」的模式，更像**系统性精度偏差**（动作量纲/数值偏移），而不是某个开关逻辑错了。

### 2.3 时序指标

| | π0.5 | SVLA（AEO 开） | SVLA（AEO 关） |
| --- | --- | --- | --- |
| Action Speed | 30.93 a/s | 33.15 a/s | 30.57 a/s |
| **T_action = 1000 / speed** | **32.3 ms** | **30.2 ms** | **32.7 ms** |
| 单步仿真耗时（均值） | 19.79 ms | 21.08 ms | 20.34 ms |
| 论文 `T_action` | — | **31.6 ms** | — |

**这是本轮唯一与论文高度吻合的指标**：SVLA 实测 30.2 ms vs 论文 31.6 ms，误差 4%。说明流式推理的调度链路（逐动作回推、动作队列、AEO 触发时机）在**结构上**是对的，只是决策内容不准。

### 2.4 AEO 行为观察

n=1 冒烟的干净样本（服务端日志 710 行）：

- 常规观测（`threshold=100000` 路径）：**43 次**
- AEO 判定「Δembedding ≤ 4.43」→ 抑制早期重规划（`Skipping denoise`）：**33 次**
- 抑制率约 **43%**

即客户端大约每 2.3 次询问就有 1 次被判定为"还不需要重规划"。这个量级合理，说明预测器在工作，但**无法判断阈值 4.43 是否与训练时标定的空间一致**（见方案 F）。

> 注：n=10 的两次运行共用了同一个服务端进程（AEO 开/关两个客户端先后连接），日志里两种模式的计数混在一起（2206 次常规观测 / 321 次抑制），无法干净拆分，故此处只用 n=1 的干净样本。

### 2.5 与论文对照

| 指标 | 论文 | 本项目实测 | 差距 |
| --- | --- | --- | --- |
| `libero_spatial` 成功率 | 94.9%（全 4 套件平均） | 42.0%（AEO 开） | **−52.9 点** |
| `T_action` | 31.6 ms | 30.2 ms | **−1.4 ms（吻合）** |

---

## 3. 差距归因分析

### 3.1 归因方法

采用排除法：先证明**外围评估环境**没有问题，再逐步收缩到**服务端内部**，最后对每个候选原因给出可证伪的判据。

### 3.2 已排除的原因（每条都附证据）

| # | 假设 | 排除依据 |
| --- | --- | --- |
| 1 | **初始状态偏难**（首轮怀疑） | π0.5 在同一批 `initial_states[0..9]` 上 99%，其中包含 SVLA 全灭的 4 个任务。若是 init state 问题，π0.5 也该失败 |
| 2 | **客户端链路错误**（渲染 / 动作应用 / 成功判定 / seed） | π0.5 与 SVLA 共用同一份 `LiberoInputs`/`LiberoOutputs` 与同一套仿真循环，π0.5 达 99% |
| 3 | **`use_sfp` 取值错误** | 检查点自带 `metadata.pt` 中作者原始配置写死 `use_sfp: True`，与本项目一致 |
| 4 | **归一化映射用错**（仿射 vs 线性分位数） | `policy_config.py:81/86` 推理路径**从不传** `use_linear_quantile`，所以无论配置里 `use_linear_quantile_norm` 是什么，推理一律走仿射分位数 `2(x−q01)/(q99−q01)−1`。训练侧虽传了该参数，但训练配置的 `model` 里没有 `svla` 字段 → 当时 `model_type` 是 `PI05` → 也是 `False`。两边一致 |
| 5 | **`action_states` 语义错误** | 训练侧 `scripts/preprocess_dataset.py:43-44`：零向量前置 + `np.cumsum` + 去掉末行，即 `action_states[t] = Σ actions[0..t-1]`；客户端 `action_states = np.zeros(7)` 后每步 `+= action`，与训练严格对应 |
| 6 | **统计量的计算方法错误** | 官方 `scripts/compute_norm_stats_new.py:113` 就是 `stats["action_states"].update(np.asarray(batch["actions"]))`，与本项目 `scripts/make_norm_stats.py` 的做法逐字一致 |
| 7 | **AEO 是主因** | 关掉 AEO 后 42% → 53%，只回收 11 个点（任务 #5 从 1/10 提到 5/10）。AEO 有损失但不是主因 |

### 3.3 剩余假设与定量估算

#### 假设 ①：发布的检查点不是论文最终模型 —— **可能性最高**

`metadata.pt` 里是作者的原始训练配置：

```
config.name   = 'pi05_libero_sfp_test'          # 名字带 _test
exp_name      = 'state_dataset'
global_step   = 58300                            # 论文说 6 万步
teacher_model = {action_horizon: 50, use_sfp: False, ...}   # 蒸馏教师
pytorch_weight_path = '/mnt/public/shiyiran/openpi_pytorch/openpi/checkpoints/pi05_libero_pytorch'
data.repo_id  = '/mnt/public/shiyiran/datasets/libero_sfp'
data.assets   = {assets_dir: None, asset_id: None}
ema_decay     = 0.999
```

- `_test` 后缀 + 未跑满 6 万步，都指向**一次实验性运行**而非最终交付权重。
- 同时 `data.assets.asset_id = None`，说明作者训练时 norm stats 是从 `libero_sfp` 目录读的（`asset_id` 回退成 `repo_id`），而这份 `assets/libero_sfp` **没有随检查点发布**。

若假设成立，则本项目的代码侧没有问题，42% 就是这份权重的真实水平。

#### 假设 ②：归一化统计量数值与原训练数据不同 —— **可能性中等偏低**

- 作者用 `/mnt/public/shiyiran/datasets/libero_sfp` 的 q01/q99，本项目用 `physical-intelligence/libero` 的。
- 仿射映射 `2(x−q01)/(q99−q01)−1` 对 q01/q99 的**量程**敏感：若 q99−q01 偏离 30%，归一化后的动作尺度就偏离 ±30%，对精细放置任务影响显著（与 2.2 的分布特征吻合）。
- 但两者都是同一批 LIBERO 演示，且 SVLA 是从 `pi05_libero_pytorch` 微调而来，统计量理应非常接近。**单靠它解释 50 个点的差距偏勉强**，不过值得用方案 B 一次性排除。

#### 假设 ③：`action_left_sum` 条件分布外推 —— **可能性低**

`svla_pytorch.py:229` 在 `action_left_sum is not None` 时会给 prefix 注入额外条件。训练数据集没有这一列，推测训练时恒为 `None`；客户端在 AEO 模式下会传入 7 维向量（剩余动作的累计量），属训练分布外。

但 AEO 关闭时该条件恒为 `None`（等价训练分布），成功率仍是 53% → **不是主因**。

#### 假设 ④：发布仓库被裁剪，仍有未知的与训练行为不一致之处 —— **可能性低但不可忽略**

本项目发布状态本身就缺 6 处**必需**代码（`pi0_fast.py`、`droid_policy.py`、`ModelType.PI0_FAST`、`ModelTransformFactory` 的 SVLA 分支、`Pi0Config.use_sfp`、AEO 预测器路径），说明作者清理仓库时删多了、且没有验证过可运行性。不能排除还有其他与训练实际行为不一致的地方。

---

## 4. 解决方案（重点）

### 4.1 方案速查表

| 方案 | 目标 | 成本 | 前置依赖 | 可给出结论 |
| --- | --- | --- | --- | --- |
| **A. 动作数值比对** | 判定推理链路是否有系统性数值偏差 | **10 分钟** | 无 | 链路对/错，直接决定走 B 还是 D/E |
| **B. 重算 norm stats** | 排除统计量差异 | 2–6 小时 + 数十 GB 磁盘 | 需下载 LIBERO 数据 | 统计量是不是主因 |
| **C. 跨套件验证** | 判定是否只 spatial 敏感 | ~1 小时 | 无 | 差距是全面性还是局部性 |
| **D. 向作者确认** | 确认权重与 assets | 沟通成本 | 无 | 最快的定性途径 |
| **E. 检查 EMA 权重** | 若发布的是非 EMA 权重 | 13.5 GB 下载 + 解析 | 无 | 换权重能否提升 |
| **F. AEO 阈值标定** | 回收 AEO 造成的 11 个点 | 每档 10 分钟 | 无 | 上限只有 53% |
| **G. 修复归一化传递隐患** | 代码健壮性 | 5 分钟 | 无 | 不影响当前结论 |

**推荐执行顺序：A → (C ∥ D) → 视结果决定 B 或 E。**

---

### 4.2 方案 A：动作数值比对（首选诊断，10 分钟）

#### 目标

判断 SVLA 输出的动作**量纲与形态**是否与 π0.5 处于同一量级。这是区分「代码侧归一化问题」与「权重本身问题」最快的实验。

#### 为什么能区分

同一条 LIBERO 观测下，两个模型的第一个动作不可能相同（不同权重），但**量级必须可比**：LIBERO 的 OSC_POSE 动作是末端位置增量（米级，典型绝对值 0.1~1.0）+ 旋转增量 + 夹爪。如果 SVLA 输出系统性偏大 2 倍以上或符号紊乱，则说明归一化的反变换参数（q01/q99）不对。

#### 具体操作

**步骤 1**：在 `examples/libero/streamingvla.py` 里，第一次拿到动作处（第 195 行 `action = action_data["actions"]` 之后）插入：

```python
if episode_actions == 0 and episode_idx == 0:      # 只在第一个 episode 的第一个动作保存
    np.save("results/dbg_svla_first_action.npy", np.asarray(action))
```

**步骤 2**：在 `examples/libero/pi05.py` 里，第一次 inference 之后（`action = client.infer(element)["actions"]` 附近）插入：

```python
if episode_idx == 0 and t == args.num_steps_wait:
    np.save("results/dbg_pi05_first_action.npy", np.asarray(action[0]))
```

**步骤 3**：两个模型各跑一次 1 个 episode（服务端切换，先 SVLA 后 π0.5，端口都 8000）：

```bash
cd /home/go2/StreamingVLA/StreamingVLA && source env.sh

# --- StreamingVLA ---
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=streamingvla_pi05_libero --policy.dir=ckpt/StreamingVLA_LIBERO/58300 > results/serve_dbg_svla.log 2>&1 &
# 另一个终端
python examples/libero/streamingvla.py --args.host 0.0.0.0 --args.port 8000 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.seed 7 \
  --args.num-steps-wait 10 --args.resize-size 224 --args.replan-steps 10 \
  --args.action-left 2 --args.threshold 4.43 --args.use-judger \
  --args.timing-output-path results/dbg_svla.txt --args.video-out-path results/dbg_svla_videos >> results/serve_dbg_svla.log 2>&1

# --- 换 π0.5（先杀上面的服务端）---
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=ckpt/pi05_libero > results/serve_dbg_pi05.log 2>&1 &
python examples/libero/pi05.py --args.host 0.0.0.0 --args.port 8000 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.seed 7 \
  --args.num-steps-wait 10 --args.resize-size 224 --args.replan-steps 10 \
  --args.timing-output-path results/dbg_pi05.txt --args.video-out-path results/dbg_pi05_videos >> results/serve_dbg_pi05.log 2>&1
```

**步骤 4**：比对

```bash
.venv/bin/python - <<'PY'
import numpy as np
a = np.load("results/dbg_svla_first_action.npy").astype(float)
b = np.load("results/dbg_pi05_first_action.npy").astype(float)
print("SVLA :", np.round(a, 3))
print("pi05 :", np.round(b, 3))
na, nb = np.linalg.norm(a), np.linalg.norm(b)
print(f"L2   : SVLA={na:.4f}  pi05={nb:.4f}  ratio={na/(nb+1e-9):.3f}")
print(f"cos  : {a@b/(na*nb+1e-9):.3f}")
print(f"逐维比值: {np.round(a/(b+1e-9), 2)}")
PY
```

#### 判定标准

| 观测 | 结论 | 下一步 |
| --- | --- | --- |
| `ratio` ∈ [0.7, 1.4] 且 `cos` > 0.8 | 量纲一致，**推理链路正确** | 走 **D（问作者）+ C（跨套件）**，并考虑 **E（EMA）** |
| `ratio` ∉ [0.5, 2.0] 或 `cos` < 0.4 | 存在**系统性数值偏差** | 立刻走 **B（重算 norm stats）** |
| 某几维为 0 / NaN / 符号固定相反 | 维度对齐或夹爪维度处理有问题 | 查 `LiberoOutputs` 的 `[:, :7]` 截断与 `action_dim=32` 的 padding |

#### 成本与风险

- 成本：约 10 分钟（含两次服务端冷启动各 ~60 s）。
- 风险：低。仅新增两个 `np.save`，不影响主流程；测完记得删掉或加开关。

---

### 4.3 方案 B：重算归一化统计量

#### 目标

用与作者训练数据**同分布**的数据重算 `state` / `actions` 的 q01/q99，排除假设 ②。

#### 两条路线

**B1（快，2 小时）**：直接用 openpi 的 `physical-intelligence/libero` LeRobot 数据集

> 注意：这条路线算出来的统计量与当前在用的一模一样（当前 `norm_stats.json` 就来自 π0.5 检查点，而它正是用该数据集训练的），所以 **B1 无法产生新信息**，只在你想验证流程时使用。

**B2（慢，4–6 小时，最接近作者）**：走 `docs/StreamingVLA_run.md` §6 的完整流程

1. 下载 `openvla/modified_libero_rlds`（HF，走 `hf-mirror.com`），这是作者所用 `libero_sfp` 的上游
2. 用 openpi 的 RLDS → LeRobot 转换脚本转成 LeRobot 格式
3. 补 `action_states` 列：
   ```bash
   .venv/bin/python scripts/preprocess_dataset.py /path/to/libero_lerobot
   ```
   （该脚本对目录下所有 `*.parquet` 递归处理，零向量前置 + cumsum，与官方一致）
4. 把 `streamingvla_pi05_libero` 配置的 `repo_id` 改成实际数据集路径
5. 计算统计量：
   ```bash
   .venv/bin/python scripts/compute_norm_stats_new.py streamingvla_pi05_libero
   ```

#### ⚠️ 已验证的坑（务必注意）

`scripts/compute_norm_stats_new.py:116` 用

```python
output_path = config.assets_dirs / data_config.repo_id
```

而本项目 `repo_id` 是**绝对路径**。`pathlib` 遇到绝对路径会**丢弃前缀**，实测结果：

```
assets_dirs = /home/go2/StreamingVLA/StreamingVLA/assets/streamingvla_pi05_libero
repo_id     = /path/to/libero_dataset_with_action_states
output_path = /path/to/libero_dataset_with_action_states     ← 直接写进数据集目录！
```

所以算完后**必须手动拷贝**，并重新派生 `action_states`：

```bash
cp <数据集目录>/norm_stats.json /tmp/libero_norm_stats.json
# 复用现有脚本把 action_states 补上，指向新文件
.venv/bin/python scripts/make_norm_stats.py \
  --src /tmp/libero_norm_stats.json \
  --dst ckpt/StreamingVLA_LIBERO/58300/assets/libero/norm_stats.json
```

（`scripts/make_norm_stats.py` 已支持 `--src/--dst`，会把 `actions` 的统计量复制为 `action_states`。）

#### 判定标准

用新统计量重跑 n=10：

| 结果 | 结论 |
| --- | --- |
| 成功率从 42% 显著提升（如 >75%） | 假设 ② 成立，统计量是主因；保留新 stats，进入正式评测 |
| 提升 < 5 个点 | 假设 ② 排除；假设 ① 基本坐实，转 D/E |

#### 成本与风险

- 成本：数据下载（数十 GB，走 HF 镜像约 25 MB/s → 20–40 分钟）+ 转换与计算（1–3 小时）。磁盘需预留 ~80 GB。
- 风险：中。转换脚本路径需自行核对；本仓库没有提供 RLDS→LeRobot 的转换脚本（只有 `preprocess_dataset.py` 和 `compute_norm_stats_new.py`），需从 openpi 借用，版本可能与作者的转换方式不同。

---

### 4.4 方案 C：跨套件验证（1 小时）

#### 目标

判断 42% 是**全面性偏低**还是**仅 spatial 套件敏感**。如果 `libero_object` / `libero_goal` / `libero_10` 也在 40–60%，说明是全局问题；如果某些套件接近 90%，则说明是精度敏感任务的特有表现。

#### 具体操作

服务端保持 StreamingVLA（AEO 开），依次跑 3 个套件，n=10：

```bash
cd /home/go2/StreamingVLA/StreamingVLA && source env.sh
for S in libero_object libero_goal libero_10; do
  python examples/libero/streamingvla.py --args.host 0.0.0.0 --args.port 8000 \
    --args.task-suite-name $S --args.num-trials-per-task 10 --args.seed 7 \
    --args.num-steps-wait 10 --args.resize-size 224 --args.replan-steps 10 \
    --args.action-left 2 --args.threshold 4.43 --args.use-judger \
    --args.timing-output-path results/timing_svla_judger_${S}_n10.txt \
    --args.video-out-path results/videos_svla_judger_${S}_n10
done
```

（可选：π0.5 同套件各跑一遍做基线，总时长翻倍。）

#### 判定标准

| 结果 | 结论 |
| --- | --- |
| 3 个套件都在 40–60% | 全局性偏差 → 支持假设 ①/② |
| 部分套件 >85% | 精度敏感任务特有 → 强烈支持假设 ② |
| `libero_10`（长时域）显著更低 | AEO 有害，转方案 F |

#### 成本与风险

- 成本：约 1 小时（每套件 ~12–18 分钟，步数上限更大：object 280 / goal 300 / libero_10 520）。
- 风险：低，纯读取现有环境。

---

### 4.5 方案 D：向作者确认（推荐并行进行）

#### 目标

用最低成本拿到权威答案，避免在假设 ①/② 上无限试错。

#### 需要问清的 4 件事

1. HF 上 `Yiran-Shi/StreamingVLA_LIBERO` 的 `58300` 是否就是论文表 1 中 94.9% 对应的权重？还是实验性检查点？
2. 能否提供训练时使用的 `assets/libero_sfp/norm_stats.json`（或直接打包进检查点）？
3. `libero_sfp` 数据集与 `openvla/modified_libero_rlds` 的对应关系与转换脚本？
4. 发布仓库缺少 `pi0_fast.py`、`droid_policy.py`、`ModelType.PI0_FAST`、`ModelTransformFactory` 的 SVLA 分支、`Pi0Config.use_sfp`、AEO 预测器路径，是否有完整可运行版本？

#### 成本与风险

- 成本：沟通时间不可控。
- 风险：低。可作为其他方案并行推进的主线。

---

### 4.6 方案 E：检查 EMA 权重（13.5 GB 下载）

#### 目标

作者配置里 `ema_decay = 0.999`。openpi 的 PyTorch 训练器可能把 EMA 影子权重存在 `optimizer.pt` 里；如果 `model.safetensors` 是**非 EMA** 的原始权重，换用 EMA 权重可能有明显提升。

#### 具体操作

```bash
cd /home/go2/StreamingVLA/StreamingVLA/ckpt/StreamingVLA_LIBERO/58300

# 下载（走 HF 镜像，约 10 分钟）
aria2c -c -x 8 -s 8 -j 1 \
  -d . -o optimizer.pt \
  "https://hf-mirror.com/Yiran-Shi/StreamingVLA_LIBERO/resolve/main/58300/optimizer.pt"

# 看里面有没有 ema / shadow 权重
cd /home/go2/StreamingVLA/StreamingVLA
.venv/bin/python - <<'PY'
import torch
ck = torch.load("ckpt/StreamingVLA_LIBERO/58300/optimizer.pt", map_location="cpu", weights_only=False)
print("top-level type:", type(ck))
if isinstance(ck, dict):
    for k, v in ck.items():
        print(f"  {k}: {type(v)}", (len(v) if hasattr(v, "__len__") else ""))
PY
```

若发现 `ema` / `shadow_params` 之类的键，就把其张量导出成 `ema.safetensors`，替换 `--policy.dir` 指向的 `model.safetensors` 重跑 n=10。

#### 判定标准

新权重 n=10 成功率显著高于 42% → 假设 ① 的变体成立（发布的是非 EMA 权重），改用 EMA 权重。

#### 成本与风险

- 成本：13.5 GB 下载（~10 分钟）+ 解析 + 一次 n=10（~12 分钟）。
- 风险：中。`optimizer.pt` 结构与权重命名需要探索；磁盘需 +13.5 GB。也可能里面根本没有 EMA，那就白跑一趟。

---

### 4.7 方案 F：AEO 阈值标定

#### 目标

AEO 打开损失了 11 个点（53% → 42%）。在**模型本身已经 53%** 的前提下，把阈值调到合适位置可回收这部分损失。

#### 具体操作

服务端不动，只改客户端 `--args.threshold`，依次扫 3.0 / 4.43 / 6.0（或先更粗：2.0 / 10.0 看方向）：

```bash
cd /home/go2/StreamingVLA/StreamingVLA && source env.sh
for T in 2.0 4.43 8.0 1e5; do
  python examples/libero/streamingvla.py --args.host 0.0.0.0 --args.port 8000 \
    --args.task-suite-name libero_spatial --args.num-trials-per-task 10 --args.seed 7 \
    --args.num-steps-wait 10 --args.resize-size 224 --args.replan-steps 10 \
    --args.action-left 2 --args.threshold $T --args.use-judger \
    --args.timing-output-path results/timing_svla_thr${T}_n10.txt \
    --args.video-out-path results/videos_svla_thr${T}_n10
done
```

`--args.threshold 1e5` 等价于永不触发（近似 AEO 关），可作为上限参考。

#### 更严谨的标定方式

从服务端日志里导出所有 `Delta norm` 的实际分布，取合适分位数作为阈值，而不是沿用论文的 4.43（论文的 4.43 是在**作者自己的归一化空间**下标定的，我们的 norm stats 不同，直接照搬未必合适）：

```bash
grep -a "Delta norm" results/serve_*.log | grep -oE "[0-9]+\.[0-9]+" | head -1   # 注意格式，取第一条数值
```

#### 判定标准

- 存在某个阈值使成功率 ≥ 53% 且 `T_action` 仍接近 31.6 ms → 采用该阈值。
- 所有阈值都不超过 53% → AEO 不是瓶颈，别再花时间，回到 A/B。

#### 成本与风险

- 成本：每档 ~12 分钟。
- 风险：低。但**上限只有 53%**，不能解决根本问题。

---

### 4.8 方案 G：修复推理端归一化传递隐患（代码健壮性，5 分钟）

#### 问题

```python
# data_loader.py:188/216（训练路径）
_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm,
                      use_linear_quantile=data_config.use_linear_quantile_norm)

# policy_config.py:81/86（推理路径）
transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)   # 少了 use_linear_quantile
```

推理端**永远不生效** `use_linear_quantile_norm`。当前这份检查点因为训练时 `model_type` 是 `PI05`（`False`），恰好一致；但如果将来用真正以 SVLA 语义训练的权重（`use_linear_quantile_norm=True`），推理端会**静默地**用错归一化，且没有任何报错。

#### 建议

**现在不要改**（改了会让当前检查点失效）。只是登记为已知风险。若将来要做，正确做法是：

```python
transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm,
                     use_linear_quantile=data_config.use_linear_quantile_norm),
```

并同时给配置加一个显式开关，把"用哪种归一化"和 `model_type` 解耦。

---

### 4.9 决策树

```
方案 A（动作数值比对，10 min）
│
├─ 量级/方向一致（ratio 0.7~1.4, cos>0.8）
│   ├─ 并行：方案 C（跨套件，1 h） + 方案 D（问作者）
│   └─ 若 C 显示部分套件 >85%  → 支持统计量假设 → 方案 B
│       若 C 显示全面 40~60%   → 支持权重假设 → 方案 E（EMA）
│
└─ 量级异常（ratio<0.5 或 >2，或 cos<0.4）
    └─ 直接方案 B（重算 norm stats），跳过 C/D/E
```

无论走哪条，**方案 F（AEO 阈值）只在模型本身已经达标后才值得做**，因为它无法突破 53% 的上限。

---

## 5. 若暂不做根因排查：完整评测的启动建议

如果决定先把 4 套件 × 50 次的完整评测跑出来作为记录（用于后续对比），按当前配置执行：

### 5.1 时间与资源

| 配置 | 单套件（n=50） | 4 套件合计 |
| --- | --- | --- |
| SVLA（AEO 开） | ~55–70 min | **~4–5 h** |
| SVLA（AEO 关） | ~55–70 min | ~4–5 h |
| π0.5 基线 | ~50–60 min | ~3.5–4 h |
| 合计 | | **~11–14 h** |

（依据：n=10 实测每轮 11–14 分钟；步数上限更大时失败 episode 更耗时。）

### 5.2 执行脚本

```bash
cd /home/go2/StreamingVLA/StreamingVLA && source env.sh
SUITES="libero_spatial libero_object libero_goal libero_10"

# ---- 终端 1：服务端（SVLA）----
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=streamingvla_pi05_libero --policy.dir=ckpt/StreamingVLA_LIBERO/58300 \
  > results/serve_full_svla.log 2>&1 &

# ---- 终端 2：客户端 ----
for S in $SUITES; do
  nohup python examples/libero/streamingvla.py --args.host 0.0.0.0 --args.port 8000 \
    --args.task-suite-name $S --args.num-trials-per-task 50 --args.seed 7 \
    --args.num-steps-wait 10 --args.resize-size 224 --args.replan-steps 10 \
    --args.action-left 2 --args.threshold 4.43 --args.use-judger \
    --args.timing-output-path results/timing_svla_judger_${S}_n50.txt \
    --args.video-out-path results/videos_svla_judger_${S}_n50 \
    > results/client_svla_judger_${S}_n50.log 2>&1
done
```

π0.5 基线只需先杀服务端、换 `--policy.config=pi05_libero --policy.dir=ckpt/pi05_libero` 并加
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`，客户端换成 `examples/libero/pi05.py` 且去掉
`--args.action-left/--args.threshold/--args.use-judger`。

### 5.3 注意事项

- 单卡只能串行：两个服务端**不能**同时运行，切换前必须 `kill` 旧进程并等端口释放。
- 磁盘：500 episode × 4 套件 × 3 配置的录像会很大，建议正式跑时关掉 `--args.video-out-path` 或只保留成功 episode 的录像。
- 指标换算：`T_action(ms) = 1000 / Action Speed`；`T_halt` 脚本不输出，需从 `norm_exceeded` 与 `action_left_sum` 日志自行推算。

---

## 6. 附录：数据来源

| 结论 | 数据来源文件 |
| --- | --- |
| π0.5 n=10 成功率 99/100 | `results/timing_pi05_n10.txt` |
| SVLA（AEO 开）42/100 | `results/timing_svla_judger_n10.txt` |
| SVLA（AEO 关）53/100 | `results/timing_svla_nojudge_n10.txt` |
| n=1 冒烟三组 | `results/timing_smoke_spatial.txt`、`timing_pi05_spatial.txt`、`timing_svla_nojudge_spatial.txt` |
| AEO 触发统计 | `results/serve_smoke.log`、`results/serve_svla_nojudge.log` |
| 作者原始训练配置 | `ckpt/StreamingVLA_LIBERO/58300/metadata.pt`（键 `config` / `global_step`） |
| 归一化映射实现 | `src/openpi/transforms.py:145-155`（`_normalize_quantile` / `_normalize_linear_quantile`） |
| 训练/推理归一化参数传递 | `src/openpi/training/data_loader.py:188,216` vs `src/openpi/policies/policy_config.py:81,86` |
| `action_states` 定义 | `scripts/preprocess_dataset.py:43-44`（训练）、`examples/libero/streamingvla.py:107,196`（推理） |
| 统计量计算方法 | `scripts/compute_norm_stats_new.py:107-113`、`scripts/make_norm_stats.py` |
