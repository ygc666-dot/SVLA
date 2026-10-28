# StreamingVLA 复现阶段总结（summary1）

- 代码仓库：`/home/go2/StreamingVLA/StreamingVLA`（git HEAD `644d356 fix: bug fix`）
- 参照文档：`docs/StreamingVLA_run.md`
- 完成内容：运行前准备 + 冒烟验证 + π0.5 基线与 StreamingVLA 的对照评测
- 尚未完成：正式 4 套件 × 50 次完整评测（受成功率异常影响，暂缓）

---

## 一、为运行 StreamingVLA 所做的准备工作

### 1.1 环境勘察结论（准备前实测）

| 项目        | 实测结果                                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| OS          | Ubuntu 20.04，`/dev/nvme0n1p2` 3.6T，可用 288G                                                                                      |
| GPU         | 2 × RTX 4090D（24564 MiB），驱动 550.54.14 / CUDA 12.4                                                                               |
| 内存        | 125 GiB                                                                                                                               |
| 默认 Python | `/home/go2/miniconda3/bin/python3` = 3.13.5（不满足 `requires-python >= 3.11`）                                                   |
| uv          | 0.8.23，缓存中已有 `cpython-3.11.13`                                                                                                |
| 系统 GL 库  | 已装 libGL / libEGL / libGLEW / libglfw；**未装 libOSMesa** → 用 `MUJOCO_GL=egl`                                             |
| 网络        | `huggingface.co` 不可达（000）；`hf-mirror.com`、`github.com`、`pypi.tuna.tsinghua.edu.cn`、`storage.googleapis.com` 均可达 |

### 1.2 目录落位

按需求约定，检查点放 `ckpt/`，第三方库放 `third_party/`（均位于仓库根下）。

```
StreamingVLA/
├── ckpt/
│   ├── StreamingVLA_LIBERO/58300/
│   │   ├── model.safetensors                  7.0 G   HF 下载
│   │   ├── metadata.pt                        3.1 K   HF 下载（含作者原始训练配置）
│   │   └── assets/libero/norm_stats.json      1.5 K   本地生成
│   ├── StreamingVLA_LIBERO_Predictor/
│   │   └── svla_predictor.pth                 6.0 G   HF 下载（AEO 预测器）
│   └── pi05_libero/                           12 G    GCS 下载（JAX ocdbt 格式）
│       ├── params/…
│       └── assets/physical-intelligence/libero/norm_stats.json
├── third_party/libero/                        427 M   本地复制
├── .venv/                                     7.7 G   服务端环境
├── examples/libero/.venv/                     4.4 G   客户端环境
├── scripts/make_norm_stats.py                         归一化统计量生成
├── env.sh                                             环境变量一键加载
└── .gitignore
```

说明：

- **LIBERO 子模块**：本仓库没有 `.gitmodules`，改从同级的 `/home/go2/StreamingVLA/openpi/third_party/libero` 直接复制（commit `f78abd6`，正是 `StreamingVLA_run.md` §4.2 推荐锁定的版本），零网络风险。含 130 个 bddl 文件 + 130 组 init_states（每任务 50 组）。
- **HF 下载**：`huggingface.co` 不可达，统一走 `hf-mirror.com`。**跳过 `58300/optimizer.pt`（13.47 GB）**，推理不需要。
- **π0.5 基线**：从 `gs://openpi-assets/checkpoints/pi05_libero` 下载全部 16 个对象（12.44 GB），用于论文表 1 对比。
- **README 声明的 `LICENSE` 文件实际不在仓库里**，从 `openpi` 复制了 Apache-2.0 版本。

### 1.3 双运行环境（相互隔离，均未污染 base 环境）

|        | 服务端 `.venv`                                                                                | 客户端 `examples/libero/.venv`                                                                                   |
| ------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Python | 3.11.13                                                                                         | 3.8.10                                                                                                             |
| 关键包 | torch 2.7.1+cu126 / transformers 4.53.2 / jax 0.5.3 / numpy 1.26.4 / flax 0.10.2 / lerobot(git) | torch **1.11.0+cu113** / numpy 1.22.4 / robosuite **1.4.1** / mujoco 3.2.3 / bddl 1.0.1 / numba 0.53.1 |

- **transformers 定制补丁**已打入服务端 venv（5 个文件：`models/siglip/{check.py,modeling_siglip.py}`、`models/paligemma/modeling_paligemma.py`、`models/gemma/{modeling_gemma.py,configuration_gemma.py}`），并通过仓库自带的 `check_whether_transformers_replace_is_installed_correctly()` 校验。
- **客户端依赖拆两步装**：本仓库 `requirements.txt` 锁 `robosuite==1.4.1`，LIBERO 官方锁 `1.4.0`，同时装会冲突。按"以本仓库 1.4.1 为准"的策略：先装本仓库 requirements，再用 `--no-deps` 补 LIBERO 侧缺失包（bddl / robomimic / hydra-core / easydict / gym / cloudpickle / future）。已校验 `torch==1.11.0+cu113` 未被 `robomimic==0.2.0` 顶掉。
- 已验证：4 个套件 × 10 任务 × 50 组 init_states 全部可正常加载。

### 1.4 代码修补清单（共 9 个文件修改 + 3 个新增）

| 文件                                          | 改动                                                                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `LICENSE`                                   | **新增**，从 openpi 复制（缺它 `uv sync` 直接报错）                                                                         |
| `pyproject.toml`                            | `[tool.uv] override-dependencies` 增加 `av==13.1.0`                                                                             |
| `src/openpi/models/model.py`                | `ModelType` 补回 `PI0_FAST = "pi0_fast"`                                                                                        |
| `src/openpi/models/pi0_config.py`           | 新增字段 `use_sfp: bool = True`                                                                                                   |
| `src/openpi/models/pi0_fast.py`             | **新增**，从 openpi 恢复（仓库漏了但被引用）                                                                                  |
| `src/openpi/policies/droid_policy.py`       | **新增**，从 openpi 恢复（同上）                                                                                              |
| `src/openpi/policies/policy.py`             | `infer()` 里 `torch.zeros_like(actions)` 改为按后端分派                                                                         |
| `src/openpi/models_pytorch/svla_pytorch.py` | `import os`；AEO 预测器路径由 `TODO` 改为 `ckpt/.../svla_predictor.pth`（可用环境变量 `SVLA_AEO_PREDICTOR_PATH` 覆盖）      |
| `src/openpi/training/config.py`             | ① `ModelTransformFactory` 的 PI05 分支并入 SVLA；② `streamingvla_pi05_libero` 增加 `assets=AssetsConfig(asset_id="libero")` |
| `scripts/serve_policy.py`                   | ① `policy.model` → `policy._model`；② SVLA 判定改为 `getattr` 安全探测（JAX 模型没有 `.config`）                         |
| `examples/libero/pi05.py`                   | `Args` 补 `host: str = "0.0.0.0"`                                                                                               |
| `scripts/make_norm_stats.py`                | **新增**，从 pi05 统计量派生 SVLA 所需的 `norm_stats.json`                                                                  |
| `env.sh`                                    | **新增**，统一 `LIBERO_CONFIG_PATH` / `PYTHONPATH` / `MUJOCO_GL` / `HF_ENDPOINT` / `XLA_PYTHON_CLIENT_MEM_FRACTION` |
| `.gitignore`                                | **新增**（原为空），忽略 `ckpt/`、`third_party/`、`.venv/`、`results/` 等                                             |

### 1.5 归一化统计量补齐

**关键发现：HF 上的检查点没有打包 `assets/`**，而 `policy_config.create_trained_policy` 会**无条件**从 `<检查点目录>/assets/<asset_id>` 加载 norm stats（`normalize.load` 找不到就抛 `FileNotFoundError`）。

而 SVLA 的 norm stats 必须同时含三个键，缺一不可：

- 输入侧 `Normalize` 走 `apply_tree` 按 key 匹配，缺 `action_states` 则该输入不会被归一化；
- 输出侧 `Unnormalize` 以 `strict=True` 调用，要求 norm stats 的每个键都出现在模型输出的 `{"state", "actions", "action_states"}` 里。

处理方式：取 π0.5 检查点自带的 `norm_stats.json`（含 `state`/`actions` 的 mean/std/q01/q99），把 `actions` 的统计量深拷贝一份命名为 `action_states`，写入 `ckpt/StreamingVLA_LIBERO/58300/assets/libero/norm_stats.json`。

依据是官方脚本 `scripts/compute_norm_stats_new.py:113`：

```python
stats["state"].update(np.asarray(batch["state"]))
stats["actions"].update(np.asarray(batch["actions"]))
stats["action_states"].update(np.asarray(batch["actions"]))   # 与 actions 共用统计量
```

即 `action_states` 的统计量本来就是用 actions 数组算的，方法层面与官方完全一致。

---

## 二、准备与运行过程中遇到的问题及修正方案

### 2.1 环境与依赖类

| # | 问题/报错                                                                         | 原因                                                                                                 | 修正方案                                                                                                                                             |
| - | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `uv sync` 失败：`License file does not exist: LICENSE`                        | 仓库未包含 `LICENSE`，但 `pyproject.toml` 里声明了 `license = {file = "LICENSE"}`              | 从 openpi 复制 Apache-2.0 的 `LICENSE`                                                                                                             |
| 2 | `av==14.4.0` 构建失败，要求 ffmpeg ≥ 7（本机 4.2.7），且镜像无 cp311 wheel     | lerobot 依赖；PyPI 上 av 14.x 没有 cp311 manylinux wheel                                             | `pyproject.toml` 用 `[tool.uv] override-dependencies` 把它降到有 wheel 的 `av==13.1.0`                                                         |
| 3 | `uv sync` 长时间卡住，实测单连接被限流到 ~160 KB/s，2 小时都下不完              | 清华源对单连接限流（aria2c 8 线程实测 37 MB/s）                                                      | 用 aria2c 预下载全部 201 个 wheel 到 `.wheels/`，再 `uv pip install --no-index --find-links .wheels -r req_local.txt` 离线安装；装完删除临时目录 |
| 4 | `import openpi.models_pytorch.gemma_pytorch` 报 `ModuleNotFoundError: pytest` | `gemma_pytorch.py` 在**模块顶层** `import pytest`（仓库 bug），而 `--no-dev` 没装 pytest | 补装 `pytest`                                                                                                                                      |
| 5 | 客户端 `ModuleNotFoundError: websockets`（后面还有 msgpack / dm-tree / tree）   | `openpi-client` 用 `--no-deps` 安装，依赖没带上                                                  | 补装 `websockets>=11.0 msgpack dm-tree tree pillow`，并校验 torch/numpy 未被改动                                                                   |

> 关于 `~/.cache/uv`（56 GB）：**不是本项目产生的**。1520 个条目里只有 479 个是本项目新增，`git-v0`/`simple-v18` 等可追溯到 2025-10-05，本机至少 10 个 uv 项目共用。uv 默认用**硬链接**把缓存文件链进 venv（实测 `libtorch_cuda.so` 在 cache 与 venv 里是同一个 inode、`links=2`），所以删除缓存**不会破坏**已装环境，但 **52.3 G 的空间也不会释放**（仍被 venv 引用），真正能回收的只有 6.5 G。建议用 `uv cache prune` 而不是 `rm -rf`。

### 2.2 仓库代码缺失/缺陷类（关键：发布代码无法直接运行）

| #  | 问题/报错                                                                                                                                  | 原因                                                                                                                                                                                      | 修正方案                                                                                              |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 6  | `ModuleNotFoundError: openpi.models.pi0_fast`                                                                                            | 仓库删掉了 `pi0_fast.py`，但 `training/misc/roboarena_config.py` 仍 import 它（config.py 顶层会导入该模块）                                                                           | 从 openpi 恢复 `pi0_fast.py`                                                                        |
| 7  | `ModuleNotFoundError: openpi.policies.droid_policy`                                                                                      | 同理，`droid_policy.py` 被删但 roboarena 仍引用                                                                                                                                         | 从 openpi 恢复 `droid_policy.py`                                                                    |
| 8  | `AttributeError: type object 'ModelType' has no attribute 'PI0_FAST'`                                                                    | `ModelType` 枚举删了 `PI0_FAST`，但 `ModelTransformFactory` 的 `match` 语句里仍有 `case ModelType.PI0_FAST`，Python 会无条件求值该 pattern                                      | 枚举补回 `PI0_FAST = "pi0_fast"`                                                                    |
| 9  | SVLA 配置走 `ModelTransformFactory` 时**返回 None**，后续 `model_transforms.inputs` 抛 `AttributeError`                        | `ModelTransformFactory` 只有 PI0 / PI05 / PI0_FAST 三个分支，没有 SVLA                                                                                                                  | SVLA 与 PI05 共用同一分支（SVLA 本身就是 pi05 派生：`pi05=True, svla=True`）                        |
| 10 | `AttributeError: 'Pi0Config' object has no attribute 'use_sfp'`                                                                          | `svla_pytorch.py:314` 引用了 `self.config.use_sfp`，但 `Pi0Config` 没有该字段                                                                                                       | 新增字段。**取值 `True`**（详见 2.3）                                                         |
| 11 | AEO 预测器路径是占位符 `TODO: Replace with your_AEO_predictor_checkpoint_path_here`                                                      | 作者留的 TODO                                                                                                                                                                             | 改为 `ckpt/StreamingVLA_LIBERO_Predictor/svla_predictor.pth`，支持 `SVLA_AEO_PREDICTOR_PATH` 覆盖 |
| 12 | 服务端启动即崩：`FileNotFoundError: norm_stats.json`                                                                                     | HF 检查点未打包 `assets/`                                                                                                                                                               | 见 1.5，生成三键 `norm_stats.json`                                                                  |
| 13 | 若不给 `asset_id`，norm stats 路径会退化成 `repo_id` 的绝对路径 `/path/to/libero_dataset_with_action_states`，路径拼接行为不可控     | `create_base_config`：`asset_id = self.assets.asset_id or repo_id`                                                                                                                    | 给 SVLA 配置加 `assets=AssetsConfig(asset_id="libero")`                                             |
| 14 | `serve_policy.py`：`AttributeError: 'Policy' object has no attribute 'model'`                                                          | `Policy` 内部字段叫 `_model`                                                                                                                                                          | 改为 `_model`，并兼容 `--record` 时的 `PolicyRecorder` 包装                                     |
| 15 | π0.5 服务端：`AttributeError: 'Pi0' object has no attribute 'config'`                                                                   | JAX 模型 `Pi0` 没有 `.config`，只有 PyTorch 模型有                                                                                                                                    | 改为 `getattr(getattr(model, "config", None), "svla", False)` 安全探测                              |
| 16 | π0.5 服务端推理崩：`TypeError: zeros_like(): argument 'input' must be Tensor, not ArrayImpl`                                            | `policies/policy.py:96` 在 `infer()` 里硬编码 `torch.zeros_like(actions)`，JAX 分支的 actions 是 `jax.Array`                                                                      | 按后端分派：`torch.zeros_like` / `jnp.zeros_like`                                                 |
| 17 | 流式推理崩：`Sizes of tensors must match except in dimension 2. Expected size 1 but got size 10 for tensor number 1`（`denoise_step`） | `embed_suffix` 里 `att_masks` 长度必须等于 action token 数。`use_sfp=False` 会生成 `action_horizon=10` 个 entry，而 `forward` 和 `sample_actions` 的 `x_t` 都是 `[B,1,D]` | `use_sfp` 必须为 `True`（`[1]` 单 token mask）                                                  |

### 2.3 `use_sfp` 取值是怎么定下来的

先用张量形状反推：`forward`（`svla_pytorch.py:336`，注释即写明 `x_t shape=[B,1,D]`）与 `sample_actions` 都是每步只去噪 1 个 action token，所以 `att_masks` 只能是 `[1]`，即 `use_sfp=True`；`False` 必崩。

后来在检查点自带的 `metadata.pt` 里找到作者**真实训练配置**，直接证实了这一点：

```python
model = {'action_dim': 32, 'action_horizon': 10, 'max_token_len': 200, 'dtype': 'bfloat16',
         'paligemma_variant': 'gemma_2b', 'action_expert_variant': 'gemma_300m',
         'pi05': True, 'discrete_state_input': False, 'use_sfp': True}
```

`use_sfp = True` ✅ 与我们的实现一致。

### 2.4 命令行接口类

| #  | 问题                                 | 说明                                                                                                                                         |
| -- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 18 | `serve_policy.py --port 8000` 报错 | tyro 的子命令机制下，**根参数必须写在子命令之前**：`--port 8000 policy:checkpoint --policy.config=... --policy.dir=...`              |
| 19 | 客户端 `--host 0.0.0.0` 报错       | 当前 tyro 0.9.2（装的是 0.9.2，文档按旧版写）要求 `--args.` 前缀：`--args.host`、`--args.task-suite-name`、`--args.no-use-judger` 等 |
| 20 | 服务端/客户端默认端口不一致          | 服务端默认 8192、客户端默认 8000，两边都必须显式指定                                                                                         |
| 21 | 不要用 `uv run`                    | `uv run` 会先 `sync`，在当前网络下会卡在限流下载上。统一用 `.venv/bin/python` 直接调用                                                 |
| 22 | 产出路径                             | `--timing_output_path` / `--video_out_path` 默认是 `your_timing_path` / `your_video_path`，必须显式指向 `results/`                 |

### 2.5 显存

- SVLA 服务端实测占用约 **8.3 GB**（模型约 7.5 GB + AEO 预测器约 0.8 GB，预测器在首次触发时被 `.to(device)` 搬到 GPU 并转 bf16）。
- π0.5 基线走 JAX，需用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` 限制预分配，否则会吃满整卡。
- 两者**不能同时跑**，切换时必须先杀进程再起新的。

---

## 三、当前运行结果与差距分析

### 3.1 冒烟（n=1/任务，`libero_spatial`，seed 7）

| 配置                   | 成功率      | Action Speed | T_action |
| ---------------------- | ----------- | ------------ | -------- |
| StreamingVLA（AEO 开） | 0.50 (5/10) | 32.00 a/s    | 31.3 ms  |
| π0.5（replan 10）     | 0.90 (9/10) | 23.11 a/s    | 43.3 ms  |

**T_action 与论文的 31.6 ms 高度吻合**（31.3 ms），说明流式推理调度与 AEO 机制的时序行为是对的。

### 3.2 n=10/任务对照（`libero_spatial`，同一批 init state 0–9，seed 7）

| 配置                              | 成功率                 | 备注                                   |
| --------------------------------- | ---------------------- | -------------------------------------- |
| **π0.5 基线**（replan 10） | **99/100 = 99%** | π0.5 论文值约 96–97%，本配置甚至更高 |
| StreamingVLA（AEO 开，论文配置）  | **42/100 = 42%** | 论文报 94.9%，差 53 个点               |
| StreamingVLA（AEO 关）            | **53/100 = 53%** | 隔离 AEO 后的纯模型质量                |

逐任务成功率：

| #  | 任务                              | SVLA（AEO 开） | SVLA（AEO 关） | π0.5 |
| -- | --------------------------------- | -------------- | -------------- | ----- |
| 1  | between the plate and the ramekin | 9/10           | 9/10           | 10/10 |
| 2  | next to the ramekin               | 1/10           | 1/10           | 10/10 |
| 3  | from table center                 | 5/10           | 8/10           | 10/10 |
| 4  | **on the cookie box**       | **0/10** | 1/10           | 10/10 |
| 5  | in the top drawer                 | 1/10           | 5/10           | 10/10 |
| 6  | on the ramekin                    | 10/10          | 9/10           | 9/10  |
| 7  | next to the cookie box            | 1/10           | 3/10           | 10/10 |
| 8  | on the stove                      | 10/10          | 10/10          | 10/10 |
| 9  | next to the plate                 | 5/10           | 6/10           | 10/10 |
| 10 | **on the wooden cabinet**   | **0/10** | 1/10           | 10/10 |

### 3.3 差距分析

**首先，评测链路本身是可信的。** π0.5 在同一批 init state、同一仿真环境、同一评测脚本下拿到 99%，说明客户端、渲染、动作执行、成功判定、随机种子全部正确。问题严格限定在 **SVLA 服务端内部**。

**其次，AEO 只解释了一小部分损失。** 关掉判定器后 42% → 53%（任务 #5 从 1/10 提到 5/10，符合"提前重规划有害长时域任务"的直觉），但主因不在 AEO。失败集中在需要**精细放置**的任务（放到 cookie box 上、木柜上只有 0–1/10），而简单任务能到 10/10，形态上更像系统性的精度偏差，而不是某个逻辑开关错了。

**已排除的原因：**

| 假设                           | 排除依据                                                                                                                                                                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 初始状态难度                   | π0.5 在同一批 init state 上 99%，同 4 个任务全对                                                                                                                                                                                                                                          |
| `use_sfp` 取值错误           | 检查点 `metadata.pt` 中作者配置明确写 `use_sfp: True`，与我们一致                                                                                                                                                                                                                      |
| 归一化模式（仿射 vs 线性）     | `policy_config.py:81/86` 推理路径**从不传** `use_linear_quantile`，所以不管配置怎么写，推理都用仿射分位数；训练侧 `data_loader.py:188/216` 虽然传了，但训练配置里 `model` 无 `svla` 字段 → `model_type` 当时是 `PI05` → `use_linear_quantile_norm=False`。两边一致 |
| `action_states` 语义错误     | 训练侧 `preprocess_dataset.py:43-44` 是零初始化的前缀和；客户端 `action_states = np.zeros(7)` + 每步 `action_states += action`，与训练完全对应                                                                                                                                       |
| 统计量的**计算方法**错误 | 官方 `compute_norm_stats_new.py:113` 就是用 actions 数组填 `action_states`，与本项目 `scripts/make_norm_stats.py` 一致                                                                                                                                                               |
| AEO 时序/流式调度              | T_action 31.3 ms vs 论文 31.6 ms                                                                                                                                                                                                                                                           |

**剩余怀疑点（按可能性排序）：**

1. **发布的检查点可能不是论文最终模型（我认为可能性最高）**

   `metadata.pt` 显示：

   ```
   config.name = 'pi05_libero_sfp_test'      # 名字里带 test
   exp_name    = 'state_dataset'
   global_step = 58300                        # 论文说训练 6 万步
   teacher_model.action_horizon = 50, use_sfp = False   # 蒸馏教师
   pytorch_weight_path = '/mnt/public/shiyiran/openpi_pytorch/openpi/checkpoints/pi05_libero_pytorch'
   data.repo_id = '/mnt/public/shiyiran/datasets/libero_sfp'
   data.assets  = {assets_dir: None, asset_id: None}
   ```

   这是一次实验性运行的痕迹（`_test` 后缀、未跑满 6 万步），与论文 94.9% 的成品权重可能不是同一份。
2. **norm stats 数值与原训练数据不同（可能性中等偏低）**

   作者用 `/mnt/public/shiyiran/datasets/libero_sfp` 的统计量，我们用 `physical-intelligence/libero` 的。仿射分位数映射 `2*(x-q01)/(q99-q01)-1` 对 q01/q99 比较敏感；但两者都是同一批 LIBERO 演示，且 SVLA 是从 `pi05_libero_pytorch` 微调而来，统计量理应非常接近，所以单靠它造成 53 个点的差距不太合理。
3. **`action_left_sum` 条件分布不一致（可能性低）**

   `embed_prefix` 在 `action_left_sum is not None` 时会给 prefix 注入额外条件。训练数据没有这一列，推测训练时恒为 `None`；客户端在 AEO 模式下会传入 7 维向量，属于训练分布外的条件。但 AEO 关掉时也是 53%，所以不是主因。
4. **仓库代码在发布时已被裁剪，其他隐性偏差未知**

   本项目发布状态本身就缺 6 处必需代码（见 2.2），说明作者清理仓库时删多了或没同步，不能排除还有其他与训练实际行为不一致的地方。

### 3.4 接下来的可选方案

| 方案                         | 内容                                                                                                                                                 | 成本                         | 能回答什么                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------- |
| **A. 输出数值比对**    | 给客户端加日志，在**同一观测**下分别记录 SVLA 与 π0.5 的首个动作块，比较量纲/形态                                                             | 约 10 分钟                   | 判断推理链路是否正确：量级接近→链路对，差距来自权重；量级差几倍→归一化统计量确实有问题 |
| **B. 重算 norm stats** | 按 `StreamingVLA_run.md` §6 的方案 B：下载 LIBERO LeRobot 数据 → `preprocess_dataset.py` 补 `action_states` → `compute_norm_stats_new.py` | 数小时（需下载数十 GB 数据） | 能否用更贴近原训练分布的统计量把成功率拉回来                                             |
| **C. 跨套件验证**      | 在 `libero_object` / `libero_goal` / `libero_10` 上各跑 n=10，看是否全面偏低                                                                   | 约 1 小时                    | 区分"系统性偏差"与"仅 spatial 套件敏感性"                                                |
| **D. 向作者确认**      | 确认 HF 上的 `StreamingVLA_LIBERO` 是否为论文 94.9% 对应的权重，以及 `libero_sfp` 数据集/`norm_stats` 是否可提供                               | 沟通成本                     | 最快的定性途径，可一并索取 `assets/libero_sfp`                                         |
| **E. 检查 EMA 权重**   | `58300/optimizer.pt`（13.47 GB，本次未下载）里可能含 EMA 影子权重，作者配置 `ema_decay=0.999`                                                    | 需再下 13.5 GB + 解析        | 若发布的是非 EMA 权重，用 EMA 权重可能明显更好                                           |
| **F. AEO 阈值标定**    | 在 4.43 附近扫阈值（如 3.0 / 4.43 / 6.0），或用实际 delta-norm 分布重新标定                                                                          | 每档约 10 分钟               | 能否把 AEO 带来的 -11 个点损失收回来（上限也就是 53%）                                   |

**建议的执行顺序**：先 A（10 分钟即可定性）→ 若 A 显示量级正常，则并行 C 与 D；若 A 显示量级异常，则优先 B。

---

## 四、当前系统状态

- 所有服务端与客户端进程均已关闭；端口 8000 未监听。
- 无残留的训练/评测进程。
- 产出目录 `results/` 下的文件：

| 文件                                                                                             | 内容                           |
| ------------------------------------------------------------------------------------------------ | ------------------------------ |
| `timing_pi05_n10.txt`                                                                          | π0.5 基线，99/100             |
| `timing_svla_judger_n10.txt`                                                                   | StreamingVLA（AEO 开），42/100 |
| `timing_svla_nojudge_n10.txt`                                                                  | StreamingVLA（AEO 关），53/100 |
| `timing_smoke_spatial.txt` / `timing_pi05_spatial.txt` / `timing_svla_nojudge_spatial.txt` | 三组 n=1 冒烟                  |
| `videos_*/`                                                                                    | 各次运行录像                   |
| `serve_*.log` / `client_*.log`                                                               | 服务端与客户端完整日志         |

## 五、复现命令速查

```bash
cd /home/go2/StreamingVLA/StreamingVLA
source env.sh

# ---- 终端 1：服务端（二选一，不可同时起）----
# StreamingVLA（PyTorch 分支）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=streamingvla_pi05_libero --policy.dir=ckpt/StreamingVLA_LIBERO/58300

# π0.5 基线（JAX 分支）
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=ckpt/pi05_libero

# ---- 终端 2：客户端 ----
# StreamingVLA（AEO 开，论文配置）
python examples/libero/streamingvla.py \
  --args.host 0.0.0.0 --args.port 8000 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 --args.seed 7 --args.num-steps-wait 10 --args.resize-size 224 \
  --args.replan-steps 10 --args.action-left 2 --args.threshold 4.43 --args.use-judger \
  --args.timing-output-path results/timing_svla_judger_spatial.txt \
  --args.video-out-path results/videos_svla_judger_spatial

# StreamingVLA（AEO 关，消融用）
#   同上，把 --args.use-judger 换成 --args.no-use-judger

# π0.5 基线
python examples/libero/pi05.py \
  --args.host 0.0.0.0 --args.port 8000 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 --args.seed 7 --args.num-steps-wait 10 --args.resize-size 224 \
  --args.replan-steps 10 \
  --args.timing-output-path results/timing_pi05_spatial.txt \
  --args.video-out-path results/videos_pi05_spatial
```

指标换算：`T_action(ms) = 1000 / Action Speed`。`T_halt` 脚本不输出，需自行从 `action_left_sum` / `norm_exceeded` 日志推算。
