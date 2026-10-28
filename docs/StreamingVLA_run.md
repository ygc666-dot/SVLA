# StreamingVLA 论文实验配置分析与复现指南

> 分析依据：论文 `docs/StreamingVLA.md` + 开源代码 `StreamingVLA/`（基于 openpi 官方 PyTorch 分支）。
> 所有结论均来自这两部分内容，未参考其他资料。

---

## 0. 结论速览（先回答最关键的问题）

**问：能否直接使用 π0.5 和 StreamingVLA 公开的检查点做实验？是否还需要额外训练？**

**答：复现 LIBERO 仿真实验不需要额外训练**，前提是拿到下面三个检查点：

| 用途                     | 检查点                                                             | 是否可直接推理                                                                                                          | 备注                                                         |
| ------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| StreamingVLA 主模型      | `https://huggingface.co/Yiran-Shi/StreamingVLA_LIBERO`           | ✅ 可以                                                                                                                 | 已由官方 π0.5-LIBERO 微调完成                               |
| AEO 自适应早期观测预测器 | `https://huggingface.co/Yiran-Shi/StreamingVLA_LIBERO_Predictor` | ⚠️ 可以，但**必须手动把本地路径填进** `src/openpi/models_pytorch/svla_pytorch.py`（代码里是 `TODO` 占位符） |                                                              |
| 基线 π0.5-LIBERO        | `gs://openpi-assets/checkpoints/pi05_libero`（openpi 官方）      | ✅ 可以                                                                                                                 | 也是 `scripts/serve_policy.py` 里的 `DEFAULT_CHECKPOINT` |

也就是说：

1. **只做评测（复现表 1 的成功率、$T_{action}$）**：下载检查点 → 填预测器路径 → 起服务端 → 跑 `examples/libero/streamingvla.py`，**不需要训练**。
2. **只有在以下情况才需要训练**：
   - 想复现消融实验（表 2：去掉 AFM / SA / NM / 换成随机 EO 等变体）；
   - 想复现表 1 中的 **RTC / SmolVLA / VLASH / 时序集成** 基线（这些代码**未开源**，需要自己实现）；
   - 想复现真实机器人实验。
3. **归一化统计量（norm stats）不需要自己算**：`create_trained_policy` 优先从**检查点目录内**的 `assets/<asset_id>`（`asset_id="libero"`）加载，而不是从 `./assets` 加载。只要 HF 检查点里打包了 `assets/libero`，直接推理即可。

---

## 1. 论文中的实验配置总表

### 1.1 基线模型与模型版本

| 项目                | 论文取值                                                                         | 代码中的对应位置                                                                                                                                  |
| ------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 基础模型            | **π0.5-LIBERO**（openpi 官方，流匹配动作专家）                            | `TrainConfig(name="streamingvla_pi05_libero")` 的 `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params")` |
| 骨干 VLM            | PaliGemma（`gemma_2b`）+ 动作专家（`gemma_300m`），即 π0.5 的标准 3B 级结构 | `Pi0Config` 默认：`paligemma_variant="gemma_2b"`、`action_expert_variant="gemma_300m"`                                                      |
| π0.5 默认配置      | `N_chunk = 10`，replan 到 5 个动作 → 预测时域 **h = 5**                 | `examples/libero/pi05.py` 中 `replan_steps: int = 5`                                                                                          |
| 公平对比基线        | **π0.5 (h = 10)**，即不重规划、一次执行完 10 个动作                       | 需把 `pi05.py` 的 `replan_steps` 改成 10（客户端 `action_chunk[: replan_steps]`）                                                           |
| StreamingVLA 时域   | 动作逐个生成，**不重规划**，H = 10                                         | `examples/libero/streamingvla.py` 中 `replan_steps: int = 10`；`Pi0Config(action_horizon=10)`                                               |
| 模型 dtype          | bfloat16                                                                         | `Pi0Config.dtype="bfloat16"`；`pytorch_training_precision="bfloat16"`                                                                         |
| 动作维度 / pad 维度 | 实际 7 维（6 位姿增量 + 1 夹爪），内部 pad 到 32                                 | `Pi0Config.action_dim=32`；`LiberoOutputs` 取回 `[:, :7]`                                                                                   |
| token 长度          | 200（π0.5 默认）                                                                | `Pi0Config.__post_init__`：`max_token_len = 200 if pi05 else 48`                                                                              |
| 状态输入方式        | 离散语言 token（`discrete_state_input=False` 在 LIBERO config 中被显式关掉）   | 两个 config 都写 `discrete_state_input=False`                                                                                                   |

### 1.2 训练超参（主模型 StreamingVLA）

| 项目             | 论文                                                       | 代码默认值                                                                                       | 说明                                                                |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 显卡             | 4×NVIDIA A800-SXM4-80GB                                   | `torchrun --nproc_per_node=<num_gpus>`                                                         | 论文说 10–20 GPU 小时                                              |
| 训练迭代         | **60,000**                                           | `num_train_steps=100_000`                                                                      | ⚠️ 不一致，见第 5 节                                              |
| 学习率           | **1e-5**                                             | `CosineDecaySchedule(warmup_steps=10_000, peak_lr=1e-5, decay_steps=1_000_000, decay_lr=1e-5)` | 等价于 warmup 1 万步后恒为 1e-5，与论文一致                         |
| batch size       | —                                                         | **16**（全局）                                                                             | `batch_size=16`                                                   |
| 优化器           | —                                                         | `AdamW(clip_gradient_norm=1.0)`                                                                |                                                                     |
| EMA              | —                                                         | `ema_decay=0.999`                                                                              |                                                                     |
| 稳定因子 k       | **k = 5**                                            | `k = 4`                                                                                        | ⚠️ 不一致，见第 5 节                                              |
| 初始方差 σ²₀  | **0.16**                                             | `sigma = 0.16`                                                                                 | 代码里直接用 σ（即 σ²=0.0256）还是 σ²，存在写法歧义，见第 5 节 |
| 时间步采样       | —                                                         | Beta(1.5, 1.0)，映射到 `[0.001, 1.0)`                                                          | `sample_time`                                                     |
| 动作空间状态     | 训练前预计算整条轨迹的 `action_states`（cumsum，初值 0） | `scripts/preprocess_dataset.py`                                                                | 保证动作空间状态与物理状态对齐（SA）                                |
| 归一化修正（NM） | 共享统计量 + 去掉偏置                                      | `use_linear_quantile_norm=True` → `x / scale`                                               | 见下节                                                              |

**归一化修正（NM）在代码中的落点**：

- `DataConfigFactory.create_base_config` 对 SVLA 模型类型设置
  `use_quantile_norm=True`（非 PI0 即可）、`use_linear_quantile_norm=True`（`model_type == ModelType.SVLA`）。
- `Normalize._normalize_linear_quantile`：`scale = max(|q99|, |q01|) + 1e-6; return x / scale`
  —— **没有减均值、没有偏置项**，正是论文式 9 描述的"去掉偏置的线性分位归一化"。
- `scripts/compute_norm_stats_new.py` 中 `stats["action_states"].update(np.asarray(batch["actions"]))`
  —— **动作空间状态与动作共用同一套统计量**，对应论文"在所有动作空间的状态与动作之间共享归一化统计量"。

### 1.3 AEO 预测器配置

| 项目           | 论文                                                           | 代码                                                                                                                                               |
| -------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 结构           | 轻量 Transformer，ViT 图像嵌入为输入，剩余动作做 DiT 条件控制  | `DiTPredictor(emb_c=2048, emb_len=256, action_dim=6, n_layers=15, n_heads=16, dim_feedforward=4096, cond_dim=256)`（`svla_pytorch.py` 中构造） |
| 训练数据       | 随机采样间隔 **1–3 个动作** 的帧对                      | `predictor/config.py`：`min_horizon=1, max_horizon=3`                                                                                          |
| 损失           | MSE（预测 `Δembedding`，与"早帧嵌入 + Δ"和"晚帧嵌入"之间） | `predictor/train.py`                                                                                                                             |
| 迭代 / 学习率  | **54,000 iter，余弦调度，初值 1e-4**                     | `predictor/config.py`：`batch_size=64, learning_rate=5e-4, min_lr=3e-5, num_epochs=20` ⚠️ 与论文不一致                                       |
| 显卡 / 时长    | 2×A800，3–4 GPU 小时                                         | `torchrun --nproc_per_node=N train.py`                                                                                                           |
| 推理判据       | `‖Δembedding‖₂ > 阈值` → 抑制早期观测                   | `svla_pytorch.sample_actions`：`delta_embedding_norm > threshold` → yield `{"norm_exceeded": True}`                                         |
| **阈值** | 在精度–延迟权衡下网格搜索                                     | `examples/libero/streamingvla.py`：`threshold: float = 4.43`                                                                                   |
| 触发时机       | 队列剩余 **n** 个动作时送检                              | `action_left: int = 2`（即 H=10 下执行到第 8 步触发，`(replan_steps - action_left) % replan_steps`）                                           |
| 开销           | 每次观测调用 1 次，8–10 ms，摊到每动作 0.8–1 ms（约 5%）     | 预测器只在 `count==1`（主视角图）且 `action_left_sum is not None` 时运行                                                                       |

### 1.4 LIBERO 仿真评测配置

| 项目             | 论文                                                                                          | 代码                                                                                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 任务套件         | LIBERO-Spatial / Object / Goal / Long，各 10 个任务                                           | `task_suite_name`：`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`                                                                         |
| 每条任务试验次数 | **50**                                                                                  | `num_trials_per_task: int = 50`                                                                                                                                     |
| 随机种子         | 固定                                                                                          | `seed: int = 7`（`np.random.seed` + `env.seed`）                                                                                                                |
| 初始状态         | 每个 episode 用不同的 init state                                                              | `initial_states[episode_idx]`                                                                                                                                       |
| 最大步数         | —                                                                                            | spatial 220 / object 280 / goal 300 / libero_10 520                                                                                                                   |
| 稳定步数         | —                                                                                            | `num_steps_wait = 10`（先执行 10 步 dummy action，之后才开始计时）                                                                                                  |
| 渲染分辨率       | —                                                                                            | 环境渲染 256×256，`LIBERO_ENV_RESOLUTION=256`；送入模型前 `resize_with_pad` 到 **224×224**                                                                |
| 输入             | agentview + eye-in-hand 双视角（翻转 180°）+ 8 维状态（eef pos + axis-angle + gripper qpos） | `streamingvla.py` / `pi05.py` 的 `element` 字典                                                                                                                 |
| 评估硬件         | **单卡 NVIDIA A800-SXM4-80GB**                                                          | 服务端 `pytorch_device` 默认 `cuda`                                                                                                                               |
| 指标             | 成功率 %、$T_{action}$（ms/动作）、$T_{halt}$（ms）                                       | 脚本输出 `Overall Success Rate`、`Avg Action Speed (actions/second)`（可换算成 $T_{action}$）；**$T_{halt}$ 脚本没有直接输出**，需要自己按第 3 节方法算 |

### 1.5 真实世界实验配置（附录 A）

| 项目 | 论文                                                                    | 代码                                                                      |
| ---- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 平台 | Franka Emika Panda（7 自由度 + 平行夹爪）+ 固定 RGB 相机，桌面抓取-放置 | **代码未开源**（仓库只有 `examples/ur5/` 的 UR5 示例，非 Franka） |
| 模型 | 基于 **π0.5-base**，动作视界 **H = 8**                     | 无对应 config（`_CONFIGS` 里没有 H=8 的 pi05_base 配置）                |
| 基线 | π0.5：H=8，重规划步长 4                                                | 无                                                                        |
| 结果 | 170.88 ms vs 271.49 ms（约 1.58×）                                     | 无                                                                        |

---

## 2. 完全按论文配置开展实验的操作步骤

下面给出"除硬件外，环境与模型配置都与论文一致"的最小可复现流程。

### 2.1 环境搭建

**A. 服务端 / 主环境（Python 3.11 + uv，openpi PyTorch 分支）**

```bash
git clone --recurse-submodules git@github.com:gen-robot/StramingVLA.git
# 若已克隆：
git submodule update --init --recursive      # 必须！third_party/libero 是子模块

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

关键版本（`pyproject.toml`）：`requires-python >= 3.11`、`torch==2.7.1`、`transformers==4.53.2`、`flax==0.10.2`、`jax[cuda12]==0.5.3`、`numpy<2.0`、lerobot 固定在 rev `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。

**B. 必须做的额外一步（transformers 补丁）**
`svla_pytorch.py` 启动时会校验 `transformers_replace` 是否装好，否则直接抛错：

```bash
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

**C. LIBERO 仿真客户端（独立 Python 3.8 venv）**

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

客户端依赖（`examples/libero/requirements.in`）：`robosuite==1.4.1`、`numpy==1.22.4`、`torch==1.11.0+cu113`、`opencv-python==4.6.0.66`、`matplotlib==3.5.3`。
即 **LIBERO 通过 git 子模块引入（LIBERO + robosuite 1.4.1 组合），仓库里没有写死 commit 号，只能用子模块当前指向的版本**。

### 2.2 获取检查点（不需要训练）

```bash
# 1) StreamingVLA 主模型
git lfs install
git clone https://huggingface.co/Yiran-Shi/StreamingVLA_LIBERO            /path/to/StreamingVLA_LIBERO
# 2) AEO 预测器
git clone https://huggingface.co/Yiran-Shi/StreamingVLA_LIBERO_Predictor  /path/to/predictor.pth
# 3) 基线 π0.5-LIBERO（openpi 官方，服务端默认会自动从 gs:// 拉取）
#    gs://openpi-assets/checkpoints/pi05_libero
```

**必须手动改的一处**（README 也明确说了）：把预测器路径写进模型定义。

```104:110:StreamingVLA/src/openpi/models_pytorch/svla_pytorch.py
        checkpoint_path = "TODO: Replace with your_AEO_predictor_checkpoint_path_here"  # e.g., "gs://openpi-assets/checkpoints/aeo_predictor/your_checkpoint.pth"
      
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
```

改为本地实际路径（例如 `/path/to/predictor.pth/best_model.pth`）。预测器权重不在主模型 `state_dict` 里（`object.__setattr__` 方式挂载），所以漏改不会报"缺键"，只会加载失败/报错。

### 2.3 评测 StreamingVLA（终端 1 + 终端 2）

**终端 2（服务端）**：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=streamingvla_pi05_libero \
    --policy.dir=/path/to/StreamingVLA_LIBERO \
    --port 8000                 # ⚠️ 服务端默认 8192，客户端默认 8000，必须对齐
```

因为 `svla=True`，`serve_policy.py` 会自动走 `streaming_websocket_policy_server`（流式推流）。

**终端 1（仿真客户端）**：

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# 四个套件分别跑：libero_spatial / libero_object / libero_goal / libero_10
python examples/libero/streamingvla.py \
    --host 0.0.0.0 --port 8000 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --seed 7 --num_steps_wait 10 --resize_size 224 \
    --replan_steps 10 --action_left 2 --threshold 4.43 --use_judger True \
    --timing_output_path results/timing_svla_spatial.txt \
    --video_out_path results/videos_svla_spatial
```

以上参数即为仓库默认值，也就是论文所用设置（H=10、n=2、阈值 4.43、每任务 50 次、seed=7）。
若 MuJoCo 报 EGL 错误，加 `MUJOCO_GL=glx`。

### 2.4 评测 π0.5 基线

**h = 10（论文主要公平对比基线）**：

```bash
# 服务端
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero --port 8000
# 客户端
python examples/libero/pi05.py --task_suite_name libero_spatial --replan_steps 10 ...
```

**replan = 5（π0.5 默认，h = 5）**：把 `--replan_steps` 设为 5（脚本默认值）。

注意：`pi05_libero` 配置 `svla=False`，服务端走普通 `websocket_policy_server`；该 checkpoint 是 openpi 官方 JAX 格式（`params/`），`create_trained_policy` 检测到目录里没有 `model.safetensors` 就会走 JAX 分支直接加载，**不需要自己转换**。

### 2.5 指标怎么得到

- **成功率**：脚本直接输出 `Overall Success Rate (All Trials)`（成功 episode / 总 episode，每任务 50 次 × 10 任务）。
- **$T_{action}$**：脚本输出 `Overall Avg Action Speed (Success Only)`，单位 actions/second，**只统计成功 episode**。换算：$T_{action}(\text{ms}) = 1000 / \text{speed}$。
  计时从"稳定步结束"开始（`episode_start_time`），到 episode 结束，除以实际执行动作数，包含仿真步进时间。
- **$T_{halt}$**：脚本**不直接输出**，需要自己补。可用服务端每帧回传的 `server_timing.infer_ms` 和 `index`（`streaming_websocket_policy_server.py` 会打包）或者客户端队列为空的等待时间来统计"两次执行之间的空闲间隔"。论文附录 C1 给出的算法是：
  - π0.5：$O_{ge}=O_{oe}=0$，$T_{halt}=T_o+T_g$；
  - StreamingVLA：按重叠时间 $O_{ge}, O_{oe}$ 扣除后计算。

### 2.6 可选：自己重新训练（仅当你要复现消融实验）

```bash
# 1) 下载原始数据（openvla/modified_libero_rlds）并转 LeRobot（fps=10, 256×256, state 8 维, action 7 维）
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/libero/data
# 2) 加入动作空间状态（cumsum，初值为 0）
python scripts/preprocess_dataset.py --dataset_dir /path/to/lerobot/data
# 3) 把 config.py 里 streamingvla_pi05_libero 的 repo_id 指向该数据集，然后算归一化统计
uv run scripts/compute_norm_stats_new.py --config-name streamingvla_pi05_libero
# 4) 训练（论文：4 卡、60000 iter、lr 1e-5）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py streamingvla_pi05_libero --exp_name svla_paper \
    --num_train_steps 60000
```

预测器训练（`src/openpi/predictor/`）：先跑 `image_embedding.py` 预抽嵌入，再改 `config.py`，最后 `torchrun --nproc_per_node=2 train.py`。

---

## 3. 论文与代码的关键差异与"坑"（按影响程度排序）

这些是"想完全对齐论文"时必须知道的：

1. **`use_sfp` 未定义（会直接报错）**
   `svla_pytorch.py` 的 `embed_suffix` 用了 `self.config.use_sfp`，但 `Pi0Config` 里**根本没有这个字段**（全仓库只有这一处引用）。推理和训练都会走到这里，必然 `AttributeError`。
   本地需要补一个字段（例如给 `Pi0Config` 加 `use_sfp: bool = False`，对应"非 SFP、走 action_horizon 的完整 mask"分支）。

   ```314:314:StreamingVLA/src/openpi/models_pytorch/svla_pytorch.py
        att_masks += [1] + ([0] * (self.config.action_horizon - 1)) if not self.config.use_sfp else [1] + ([0] * (1 - 1))
   ```
2. **AEO 预测器路径是 `TODO` 占位符**（见 2.2），不填无法加载预测器。
3. **稳定因子 k 不一致**：论文 `k=5`，代码 `k=4`；论文写 `σ²₀=0.16`，代码写 `sigma = 0.16`。
   **只影响训练，不影响推理**（`sample_actions` 里不采样噪声、不用 k）。用官方发布的检查点推理时不受影响。
4. **训练步数不一致**：论文 60,000 iter，代码默认 `num_train_steps=100_000`。若要严格复现论文训练，命令行加 `--num_train_steps 60000`。
5. **预测器配置不一致**：论文 54,000 iter / lr 初值 1e-4 余弦；`predictor/config.py` 是 `batch_size=64, lr=5e-4, min_lr=3e-5, num_epochs=20`，且架构参数（`n_layers=12, hidden_dim=1024, dim_feedforward=3072`）与 `svla_pytorch.py` 里实际构造的预测器（`n_layers=15, dim_feedforward=4096`）**不匹配**。若重训预测器，必须以 `svla_pytorch.py` 的构造参数为准。
6. **服务端/客户端默认端口不一致**：`serve_policy.py` 默认 `--port 8192`，两个客户端脚本默认 `--port 8000`。必须显式对齐。
7. **`examples/libero/pi05.py` 的 `Args` 里没有 `host` 字段**，但代码里用了 `args.host`（`WebsocketClientPolicy(args.host, args.port)`）。直接跑会报 `AttributeError`，需要在 `Args` 中补 `host: str = "0.0.0.0"`。
8. **README 里 `pi0_aloha_streamingvla_pi05_liberosim` 这个 config 名在 `_CONFIGS` 中不存在**（多卡训练示例命令引用了它）。`_CONFIGS` 实际只有 `pi05_libero`、`streamingvla_pi05_libero`、若干 `debug*`、`roboarena` 配置。
9. **`compute_norm_stats_new.py` 的输出路径与加载路径不匹配**：
   保存用 `config.assets_dirs / data_config.repo_id`，而加载用 `assets_dirs / asset_id`（SVLA 配置的 `asset_id="libero"`）。自己训练时需要把产物放到 `assets/<config_name>/libero/`，或直接依赖检查点内的 `assets/libero`。
10. **评估脚本只统计"成功 episode"的时间**（`Success-Only Metrics`），论文指标口径一致但要注意失败 episode 不计入速度统计。
11. **表 1 中的 RTC / SmolVLA / VLASH / 时序集成 基线代码未开源**，仓库里只有 `pi05.py`（vanilla π0.5）和 `streamingvla.py`。要复现这几个基线需要自己实现（VLASH 还需要额外微调）。
12. **真实世界 Franka 实验代码未开源**，且 `H=8` 的 π0.5-base / StreamingVLA 配置在 `_CONFIGS` 中不存在。仓库提供的真机示例是 `examples/ur5/`（不同机械臂、不同动作空间）。
13. **只评估、不训练时不需要下载 LIBERO 数据集**；只有重训才需要 `openvla/modified_libero_rlds` 原始数据（约 30 分钟转换）。

---

## 4. 补充：`third_party/LIBERO` 子模块的安装（是否需要、装哪个版本）

> 背景：你拿到的这份源码里**没有 `third_party/` 目录，也没有 `.gitmodules`**（大概率是 zip 下载或 clone 时未带子模块），所以子模块不会自动拉取。

### 4.1 是否需要手动安装？

需要，但**只在跑 LIBERO 仿真评测时需要**。

`third_party/libero` 是 git 子模块、不是 pip 包，`examples/libero/streamingvla.py` 与 `pi05.py` 直接 `from libero.libero import benchmark / get_libero_path / OffScreenRenderEnv`。

| 你要做的事                   | 是否需要 LIBERO                      |
| ---------------------------- | ------------------------------------ |
| 跑 LIBERO 仿真评测（客户端） | ✅ 必须                              |
| 启动 policy 服务端 / 推理    | ❌ 不需要                            |
| 训练模型                     | ❌ 不需要（需要的是 LeRobot 数据集） |

### 4.2 装哪个版本

子模块来源为官方仓库 `https://github.com/Lifelong-Robot-Learning/LIBERO.git`（可由 `examples/libero/Dockerfile` 中指向 `LIBERO/blob/master/...` 的注释佐证）。

上游 openpi 锁定的子模块 commit 为 **`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`**。
（StreamingVLA 是 openpi 的 fork，其自身 pin 无法核实——`gen-robot/StramingVLA` 目前访问为 404，可能尚未公开或已改名。）

```bash
cd StreamingVLA
# A. 严格对齐 openpi 锁定版本（推荐）
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
cd third_party/libero && git checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c && cd ../..

# B. 直接用 master：当前 HEAD = 8f1084e（2025-03-15），只改了数据集下载脚本
#    （README / benchmark_scripts/download_libero_datasets.py / download_utils.py），
#    不涉及 benchmark 与 env，对评测结果无影响
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
```

### 4.3 依赖版本：两个 requirements 文件会冲突

README 的命令是同时 sync 两个文件：

```bash
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
```

但二者对 `robosuite` 的锁定**不一致**：

| 来源                                                  | 关键锁定                                                                                                                                                                                                        |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 本仓库 `examples/libero/requirements.txt`（已编译） | `robosuite==1.4.1`、`numpy==1.22.4`、`torch==1.11.0+cu113`、`mujoco==3.2.3`、`opencv-python==4.6.0.66`、`matplotlib==3.5.3`                                                                         |
| LIBERO 官方 `requirements.txt`                      | `robosuite==1.4.0`、`bddl==1.0.1`、`robomimic==0.2.0`、`hydra-core==1.2.0`、`easydict==1.9`、`gym==0.25.2`、`transformers==4.21.1`、`wandb==0.13.1`、`cloudpickle==2.1.0`、`future==0.18.2` |

**以本仓库的 `robosuite==1.4.1` 为准**（openpi 编译锁定、实际跑通的组合）。若 `uv pip sync` 因冲突报错，拆成两步：

```bash
uv pip sync examples/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install --no-deps bddl==1.0.1 robomimic==0.2.0 hydra-core==1.2.0 easydict==1.9 \
    gym==0.25.2 cloudpickle==2.1.0 future==0.18.2
uv pip install -e third_party/libero --no-deps
uv pip install -e packages/openpi-client
```

额外两个坑：

- `robomimic==0.2.0` 可能顶掉 torch，装完务必确认 `torch==1.11.0+cu113` 未被替换；
- `numba==0.53.1`（要求 numpy<1.21）与 `numpy==1.22.4` 理论冲突，这是 openpi 已编译的既有组合，照装即可。

### 4.4 装完必须做：`LIBERO_CONFIG_PATH`

否则首次 `import libero` 会**交互式要求手动输入路径，脚本直接卡死**。照 `examples/libero/Dockerfile` 的写法配置：

```bash
mkdir -p /tmp/libero && cat > /tmp/libero/config.yaml <<'EOF'
benchmark_root: <绝对路径>/StreamingVLA/third_party/libero/libero/libero
bddl_files:     <绝对路径>/StreamingVLA/third_party/libero/libero/libero/bddl_files
init_states:    <绝对路径>/StreamingVLA/third_party/libero/libero/libero/init_files
datasets:       <绝对路径>/StreamingVLA/third_party/libero/libero/datasets
assets:         <绝对路径>/StreamingVLA/third_party/libero/libero/libero/assets
EOF
export LIBERO_CONFIG_PATH=/tmp/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

`bddl_files` 与 `init_files`（每任务 50 组初始状态，正好对应 `--num_trials_per_task 50`）都随仓库提供，**评测无需额外下载任何数据**；只有在重新训练时才需要另下 `openvla/modified_libero_rlds`。

### 4.5 Windows 注意事项

本机为 win32，而 robosuite 1.4.1 + mujoco 3.2.3 + numba/llvmlite 0.36 + glfw 在原生 Windows 上基本无法顺利安装。仓库本身提供了 Linux 化方案，建议二选一：

- **WSL2（Ubuntu 22.04）+ CUDA**，按上面步骤安装；
- **Docker Desktop**：`docker compose -f examples/libero/compose.yml up --build`，`examples/libero/Dockerfile` 已配好 `LIBERO_CONFIG_PATH`、`MUJOCO_GL=egl` 及全部系统依赖。

若 MuJoCo 报 EGL 错误，按 README 使用 `MUJOCO_GL=glx python examples/libero/streamingvla.py`。

---

## 5. 一句话总结的"最小复现路径"

> 装好 uv 环境 + 打上 transformers 补丁 → `git submodule update --init --recursive` 拿到 LIBERO → 下载 `Yiran-Shi/StreamingVLA_LIBERO` 与 `..._Predictor` → 把预测器路径填进 `svla_pytorch.py` → `scripts/serve_policy.py policy:checkpoint --policy.config=streamingvla_pi05_libero --policy.dir=... --port 8000` → `python examples/libero/streamingvla.py --task_suite_name libero_{spatial,object,goal,10}` → 收集成功率与 $T_{action}$，自行补测 $T_{halt}$。
> **整个过程不需要训练任何模型**；只有在复现消融实验或表 1 的其他基线时才需要训练/自研代码。