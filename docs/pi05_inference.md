# π₀.₅ (pi05) 模型推理运行指南

本文基于 `/home/go2/StreamingVLA/openpi`（Physical Intelligence 官方 openpi 仓库）的源码，详细说明**从环境配置到实际跑通 π₀.₅ 推理**的完整流程。

> 约定：下文用 `$OPENPI_ROOT` 指代 openpi 仓库根目录，即 `/home/go2/StreamingVLA/openpi`。
> 所有命令默认在 `$OPENPI_ROOT` 下执行。

---

## 目录

1. [环境与硬件要求](#1-环境与硬件要求)
2. [安装](#2-安装)
3. [模型权重（Checkpoints）](#3-模型权重checkpoints)
4. [配置系统详解](#4-配置系统详解)
5. [仓库内已有的 pi05 配置清单](#5-仓库内已有的-pi05-配置清单)
6. [推理方式一：Python API 直接调用](#6-推理方式一python-api-直接调用)
7. [推理方式二：Policy Server + WebSocket 客户端](#7-推理方式二policy-server--websocket-客户端)
8. [推理方式三：无机器人冒烟测试](#8-推理方式三无机器人冒烟测试)
9. [不同机器人平台的输入数据格式](#9-不同机器人平台的输入数据格式)
10. [使用自己微调的 checkpoint 推理](#10-使用自己微调的-checkpoint-推理)
11. [PyTorch 后端推理](#11-pytorch-后端推理)
12. [Docker 方式运行](#12-docker-方式运行)
13. [关键参数与性能调优](#13-关键参数与性能调优)
14. [常见问题排查](#14-常见问题排查)

---

## 1. 环境与硬件要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| OS | Ubuntu 22.04（官方仅测过此版本） | 其他系统不保证可用 |
| GPU（推理） | > 8 GB 显存，如 RTX 4090 | 单卡即可 |
| Python | >= 3.11 | 仓库 `.python-version` 固定为 `3.11` |
| 包管理器 | [uv](https://docs.astral.sh/uv/) | 已安装于 `/home/go2/.local/bin/uv` |
| 系统 CUDA | **不需要** | CUDA 库由 uv 随 `jax[cuda12]` / `torch` 一起安装，系统级 CUDA 反而可能冲突 |

本机实测环境：2 × NVIDIA GeForce RTX 4090 D（24 GB），满足推理与 LoRA 微调需求。

---

## 2. 安装

### 2.1 拉取子模块

openpi 依赖 `third_party/` 下的 LeRobot、LIBERO 等子模块，**必须**初始化：

```bash
cd $OPENPI_ROOT
git submodule update --init --recursive
```

### 2.2 用 uv 建环境

```bash
cd $OPENPI_ROOT
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

- `GIT_LFS_SKIP_SMUDGE=1` 是拉取 LeRobot 依赖所必需的（跳过 LFS 大文件）。
- 之后所有命令都用 `uv run ...` 前缀执行，即可自动使用该虚拟环境。

### 2.3 验证安装

```bash
uv run python -c "import openpi, jax; print(openpi.__file__); print(jax.__version__)"
```

---

## 3. 模型权重（Checkpoints）

### 3.1 官方 pi05 相关权重

| 权重 | 用途 | 路径 |
| --- | --- | --- |
| π₀.₅ base | 微调基座（**不直接做零样本推理**） | `gs://openpi-assets/checkpoints/pi05_base` |
| π₀.₅-DROID | DROID 平台开箱推理 / 继续微调 | `gs://openpi-assets/checkpoints/pi05_droid` |
| π₀.₅-LIBERO | LIBERO 基准开箱推理 | `gs://openpi-assets/checkpoints/pi05_libero` |

### 3.2 下载与缓存机制

**无需手动下载**。代码中凡调用 `openpi.shared.download.maybe_download(...)` 或 `create_trained_policy(..., checkpoint_dir)`，都会自动把 `gs://` 路径下载到本地缓存：

- 默认缓存目录：`~/.cache/openpi/`（结构为 `~/.cache/openpi/openpi-assets/checkpoints/<name>/`）
- 覆盖缓存目录：设置环境变量 `OPENPI_DATA_HOME`

```bash
export OPENPI_DATA_HOME=/data/openpi_cache   # 可选
```

checkpoint 目录结构（JAX 版）：

```
pi05_droid/
├── params/                 # orbax 权重
└── assets/
    └── droid/
        └── norm_stats.json # 归一化统计量，推理时必需
```

> 本机 `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid` 已下载完毕，可直接离线跑 DROID 推理。

> 注意：`download.py` 中有一段缓存失效逻辑（`_INVALIDATE_CACHE_DIRS`），对 `openpi-assets/checkpoints/` 下的旧缓存会按 mtime 判定过期并重新下载。

---

## 4. 配置系统详解

pi05 推理的配置由三个层次组成，全部定义在 `src/openpi/training/config.py` 与 `src/openpi/models/pi0_config.py`。

### 4.1 模型配置：`Pi0Config`

π₀.₅ 与 π₀ 共用同一个类，通过 `pi05=True` 区分（`src/openpi/models/pi0_config.py`）：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"       # VLM 主干
    action_expert_variant: _gemma.Variant = "gemma_300m" # 动作专家

    action_dim: int = 32        # pi05 统一用 32 维动作空间
    action_horizon: int = 50    # 一次预测的 action chunk 长度
    max_token_len: int = None   # pi05 自动设为 200（pi0 为 48）

    pi05: bool = False                 # ← 关键开关
    discrete_state_input: bool = None  # pi05 默认 True
    pytorch_compile_mode: str | None = "max-autotune"
```

π₀.₅ 相对 π₀ 的两点差异（源码注释）：
1. **state 作为离散语言 token 输入**，而不是拼在 suffix 里的连续输入（`discrete_state_input=True`）；
2. **动作专家使用 adaRMSNorm** 注入 flow matching 的 timestep。

> 本仓库目前**只支持 flow matching 头**，不支持 π₀.₅ 的自回归（FAST）头。

### 4.2 数据配置：`DataConfigFactory`

决定了「原始观测 dict → 模型输入」和「模型输出 → 动作」的映射。推理时用到的关键字段：

| 字段 | 作用 |
| --- | --- |
| `repack_transforms` | 仅在**读数据集**时生效，推理时不生效 |
| `data_transforms` | 机器人特有变换，**训练与推理都生效**（如 `DroidInputs` / `LiberoInputs` / `AlohaInputs`） |
| `model_transforms` | 模型通用变换：注入默认 prompt、resize 到 224×224、tokenize prompt、pad state/action 到 `action_dim` |
| `assets.asset_id` | 决定从 checkpoint 的 `assets/<asset_id>/norm_stats.json` 加载哪套归一化统计量 |
| `use_quantile_norm` | pi05/pi0-FAST 为 `True`（分位数归一化），pi0 为 `False` |
| `prompt_from_task` | 数据集场景下从 LeRobot 的 `task` 字段取 prompt |

归一化统计量（`norm_stats`）的加载优先级（`policy_config.create_trained_policy`）：
1. 若显式传入 `norm_stats=` 参数 → 用它；
2. 否则从 **checkpoint 目录**下的 `assets/<asset_id>` 加载（保证与训练一致）；
3. `asset_id` 为 `None` 时直接报错 `Asset id is required to load norm stats.`

### 4.3 训练配置：`TrainConfig`

推理时同样用 `TrainConfig`（名字叫 Train，但推理也靠它），核心字段：

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str                       # 配置名，get_config() 用
    model: BaseModelConfig          # 上面的 Pi0Config
    data: DataConfigFactory         # 上面的数据配置
    weight_loader: WeightLoader     # 训练用；推理不加载
    pytorch_weight_path: str|None   # PyTorch 权重路径
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    policy_metadata: dict|None = None  # 会随 policy server 的 metadata 下发给客户端
```

通过名字取配置：

```python
from openpi.training import config as _config
config = _config.get_config("pi05_droid")   # 名字写错会提示相近的候选名
```

---

## 5. 仓库内已有的 pi05 配置清单

全部位于 `src/openpi/training/config.py` 的 `_CONFIGS` 列表：

| 配置名 | 模型配置 | 数据配置 / asset_id | 用途 |
| --- | --- | --- | --- |
| `pi05_droid` | `Pi0Config(action_horizon=15, pi05=True)` | `SimpleDataConfig`，`asset_id="droid"` | **DROID 开箱推理**（推荐首选） |
| `pi05_libero` | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` | `LeRobotLiberoDataConfig`，`repo_id="physical-intelligence/libero"` | **LIBERO 开箱推理 / 微调** |
| `pi05_aloha` | `Pi0Config(pi05=True)` | `LeRobotAlohaDataConfig`，`asset_id="trossen"` | ALOHA 平台（配合 `pi05_base` 权重） |
| `pi05_aloha_pen_uncap` | `Pi0Config(pi05=True)` | `LeRobotAlohaDataConfig`，`repo_id="physical-intelligence/aloha_pen_uncap_diverse"` | ALOHA 拔笔帽微调示例 |
| `pi05_droid_finetune` | `Pi0Config(pi05=True, action_dim=32, action_horizon=16)` | `LeRobotDROIDDataConfig` | 自定义小数据集 DROID 微调 |
| `pi05_full_droid_finetune` | `Pi0Config(pi05=True, action_dim=32, action_horizon=16)` | RLDS DroidDataConfig | 全量 DROID 微调 |
| `debug_pi05` | `Pi0Config(pi05=True, paligemma_variant="dummy", ...)` | `FakeDataConfig` | 无需权重的冒烟测试 |

查看全部可用配置名：

```bash
uv run python -c "
from openpi.training import config as c
print([x.name for x in c._CONFIGS if 'pi05' in x.name])
"
```

---

## 6. 推理方式一：Python API 直接调用

适合：在同一进程里直接拿到动作（仿真评测、离线批量推理、自己的机器人运行时）。

### 6.1 最小示例（DROID / π₀.₅-DROID）

把下面内容保存为 `$OPENPI_ROOT/run_pi05_droid.py`：

```python
import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

# 1) 取配置
config = _config.get_config("pi05_droid")

# 2) 取权重（gs:// 会自动下载并缓存到 ~/.cache/openpi，本地路径直接返回）
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 3) 创建 policy（内部完成：加载权重 → 组装 transforms → 加载 norm_stats）
policy = policy_config.create_trained_policy(
    config,
    checkpoint_dir,
    default_prompt="pick up the fork",   # 观测里没有 prompt 时的兜底指令
)

# 4) 构造观测并推理
example = {
    "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left":      np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/joint_position":        np.random.rand(7).astype(np.float64),
    "observation/gripper_position":      np.random.rand(1).astype(np.float64),
    "prompt": "pick up the fork",
}

result = policy.infer(example)
actions = result["actions"]        # numpy, shape (15, 8)  = (action_horizon, 8)
print("actions:", actions.shape)
print("policy_timing:", result["policy_timing"])   # {"infer_ms": ...}
```

运行：

```bash
cd $OPENPI_ROOT
uv run python run_pi05_droid.py
```

### 6.2 `create_trained_policy` 完整签名

```python
policy_config.create_trained_policy(
    train_config: TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,  # 最先执行的额外输入变换
    sample_kwargs: dict | None = None,                  # 透传给 model.sample_actions，如 {"num_steps": 10}
    default_prompt: str | None = None,                  # prompt 兜底
    norm_stats: dict | None = None,                     # 手动指定归一化统计量
    pytorch_device: str | None = None,                  # PyTorch 后端用，如 "cuda:0" / "cpu"
) -> Policy
```

**JAX / PyTorch 自动识别**：函数会检查 `checkpoint_dir/model.safensors` 是否存在——存在即走 PyTorch 分支，否则走 JAX 分支。

### 6.3 `Policy.infer` 说明

```python
def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict
```

- 内部流程：`输入 transforms（repack → 注入 prompt → 机器人变换 → 归一化 → 模型变换）` → `sample_actions` → `输出 transforms（反归一化 → 机器人输出变换 → 截取有效维度）`
- 返回值：`{"state": ..., "actions": (H, D) ndarray, "policy_timing": {"infer_ms": float}}`
- `noise` 可选，用于固定 flow matching 的初始噪声以复现实验。

---

## 7. 推理方式二：Policy Server + WebSocket 客户端

适合：**机器人端算力弱 / 环境依赖冲突**时，把模型放到有 GPU 的服务器上，机器人端通过 WebSocket 拉动作（官方推荐）。

### 7.1 启动服务端

```bash
cd $OPENPI_ROOT

# 显式指定 config + checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

# 等价简写（DEFAULT_CHECKPOINT 里 DROID -> pi05_droid）
uv run scripts/serve_policy.py --env=DROID
```

`--env` 的默认值映射（`scripts/serve_policy.py`）：

| `--env` | config | checkpoint |
| --- | --- | --- |
| `DROID` | `pi05_droid` | `gs://openpi-assets/checkpoints/pi05_droid` |
| `LIBERO` | `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` |
| `ALOHA` | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` |
| `ALOHA_SIM` | `pi0_aloha_sim` | `gs://openpi-assets/checkpoints/pi0_aloha_sim` |

服务端可选参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--port` | `8000` | 监听端口 |
| `--default-prompt` | `None` | 观测里没有 `prompt` 时的兜底指令 |
| `--record` | `False` | 把每步输入/输出落盘到 `policy_records/` 便于调试 |

### 7.2 客户端调用

在自己的机器人运行时中嵌入客户端（依赖极少的 `openpi-client` 包）：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="<server_ip>", port=8000)

observation = {
    "observation/exterior_image_1_left": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(exterior_img, 224, 224)
    ),
    "observation/wrist_image_left": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    ),
    "observation/joint_position": joint_pos,      # (7,)，无需归一化，服务端会做
    "observation/gripper_position": gripper_pos,  # (1,)
    "prompt": "pick up the fork",
}

# 返回 action chunk，shape (action_horizon, action_dim)，例如 (15, 8)
action_chunk = client.infer(observation)["actions"]
```

要点：
- **图片在客户端就 resize 到 224×224 并转 uint8**，减少带宽与时延；
- **状态量传原始值**，归一化在服务端用 checkpoint 里的 `norm_stats.json` 完成；
- 通常**每 N 步才调一次 policy**，其余步开环执行 chunk 内剩余动作。

---

## 8. 推理方式三：无机器人冒烟测试

仓库自带一个发送随机观测、统计推理时延的客户端，用来验证服务是否跑通。

终端 1（服务端）：

```bash
cd $OPENPI_ROOT
uv run scripts/serve_policy.py --env=DROID
```

终端 2（客户端）：

```bash
cd $OPENPI_ROOT
uv run examples/simple_client/main.py --env DROID --host 127.0.0.1 --port 8000 --num-steps 20
```

可加 `--timing-file timing.parquet` 把时延统计落盘。
`--env` 支持 `DROID / LIBERO / ALOHA / ALOHA_SIM`，脚本内置了对应的随机观测构造函数（见 `examples/simple_client/main.py`）。

---

## 9. 不同机器人平台的输入数据格式

数据格式由各 `*_policy.py` 中的 `Inputs` 类决定。PaliGemma 侧固定接收三个图像槽：`base_0_rgb`（第三人称）、`left_wrist_0_rgb`、`right_wrist_0_rgb`；缺失的槽用全零图 + `image_mask=False` 占位。

### 9.1 DROID（`pi05_droid`，`src/openpi/policies/droid_policy.py`）

```python
{
    "observation/exterior_image_1_left": (224, 224, 3) uint8,   # 外部相机
    "observation/wrist_image_left":      (224, 224, 3) uint8,   # 腕部相机
    "observation/joint_position":        (7,) float,            # 关节角
    "observation/gripper_position":      (1,) float,            # 夹爪
    "prompt": "pick up the fork",                               # str
}
```

- state = `concat(joint_position, gripper_position)` → 8 维
- 输出：`{"actions": (15, 8)}`（`DroidOutputs` 截取前 8 维；`action_horizon=15`）
- 生成随机样例：`droid_policy.make_droid_example()`

### 9.2 LIBERO（`pi05_libero`，`src/openpi/policies/libero_policy.py`）

```python
{
    "observation/state":       (8,) float,
    "observation/image":       (224, 224, 3) uint8,   # 第三人称
    "observation/wrist_image": (224, 224, 3) uint8,   # 腕部
    "prompt": "pick up the alphabet soup",
}
```

- 输出：`{"actions": (10, 7)}`（`LiberoOutputs` 截取前 7 维；`action_horizon=10`）
- 生成随机样例：`libero_policy.make_libero_example()`
- 官方 Docker 化评测（含 policy server + LIBERO 环境）：

```bash
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
# 指定任务套件
CLIENT_ARGS="--args.task-suite-name libero_10" SERVER_ARGS="--env LIBERO" \
    docker compose -f examples/libero/compose.yml up --build
```

### 9.3 ALOHA（`pi05_aloha`，`src/openpi/policies/aloha_policy.py`）

```python
{
    "state": (14,) float,           # [左臂6+夹爪1, 右臂6+夹爪1]
    "images": {
        "cam_high":        (3, 224, 224) uint8,   # 必需（作为 base_0_rgb）
        "cam_low":         (3, 224, 224) uint8,   # 可选
        "cam_left_wrist":  (3, 224, 224) uint8,   # 可选
        "cam_right_wrist": (3, 224, 224) uint8,   # 可选
    },
    "prompt": "fold the towel",
}
```

- 图片通道顺序为 `(C, H, W)`，代码内部会转成 `(H, W, C)`
- 注意 ALOHA 会做「Aloha 空间 ↔ π 内部空间」的关节翻转与夹爪角度换算（`adapt_to_pi=True`）
- 输出：`{"actions": (50, 14)}`（默认 `action_horizon=50`，截取前 14 维）
- 生成随机样例：`aloha_policy.make_aloha_example()`

### 9.4 真机部署提示

- DROID 真机步骤见 `examples/droid/README.md`：把 `examples/droid/main.py` 拷到 DROID 的 `scripts/` 目录，改相机 ID，然后
  `python3 scripts/main.py --remote_host=<server_ip> --remote_port=8000 --external_camera="left"`
- ALOHA 真机见 `examples/aloha_real/README.md`，服务端：
  `uv run scripts/serve_policy.py --env ALOHA --default-prompt='take the toast out of the toaster'`

---

## 10. 使用自己微调的 checkpoint 推理

完整链路（以 LIBERO 为例，其它数据集同理）：

### 10.1 准备 LeRobot 格式数据

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

自定义数据可照此脚本改写（DROID 用 `examples/droid/convert_droid_data_to_lerobot.py`）。

### 10.2 在 `config.py` 中定义/复用配置

至少确认三处：
- `model`：`Pi0Config(pi05=True, action_horizon=..., action_dim=32)`
- `data`：自定义 `DataConfigFactory`（重点是 `Inputs`/`Outputs` 与 `assets.asset_id`）
- `weight_loader`：`CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`

### 10.3 计算归一化统计量（必需）

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 10.4 训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

产物在 `checkpoints/pi05_libero/my_experiment/<step>/`，其中会带 `params/` 与 `assets/`。

### 10.5 起服务推理

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

**关键约束**：本地 checkpoint 目录下必须存在 `assets/<asset_id>/norm_stats.json`，否则 `create_trained_policy` 会因加载不到统计量而失败。
（`asset_id` = `AssetsConfig.asset_id` 或 `repo_id`；`pi05_libero` 为 `physical-intelligence/libero`。）

---

## 11. PyTorch 后端推理

openpi 已提供 π₀ / π₀.₅ 的 PyTorch 实现（不支持 π₀-FAST、混合精度训练、FSDP、LoRA、EMA）。

### 11.1 打 transformers 补丁（**必需**）

```bash
cd $OPENPI_ROOT
uv sync
uv pip show transformers        # 必须是 4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

补丁作用：支持 AdaRMS、控制激活精度、允许 KV cache 不被更新。
> 警告：uv 默认是硬链接模式，此操作会**永久影响 uv 缓存中的 transformers**，可能波及其它项目。彻底撤销需执行 `uv cache clean transformers`。

### 11.2 把 JAX 权重转成 PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
```

- `--config_name` 用于取 `Pi0Config`（`pi05=True` 与否会影响 `time_mlp_*` / `state_proj` 的转换分支）
- 转换脚本内部还会用 `"pi05" in checkpoint_dir` 判断是否走 adaRMS 分支——**输出/输入路径中带 `pi05` 字样时才正确**，建议输出目录名保留 `pi05`
- 可选 `--precision bfloat16|float32|float16`（默认 bfloat16）
- 产物：`model.safetensors` + `config.json` + 拷贝过来的 `assets/`

### 11.3 推理（API 与 JAX 完全一致）

```python
from openpi.policies import policy_config
from openpi.training import config as _config

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/pi05_droid_pytorch"   # 含 model.safetensors

policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device="cuda:0")
actions = policy.infer(example)["actions"]
```

或起服务：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/pi05_droid_pytorch
```

> `pytorch_device` 不指定时，自动选 `cuda`（可用）或 `cpu`。
> `Pi0Config.pytorch_compile_mode` 默认 `"max-autotune"`，首次推理会触发 `torch.compile` 编译（较慢），之后速度与 JAX 相当；如想跳过编译可设为 `None`。

---

## 12. Docker 方式运行

前提：Docker 以 **rootless 模式**安装，且装好 **NVIDIA Container Toolkit**（用 snap 或 Docker Desktop 安装的版本不兼容）。

```bash
cd $OPENPI_ROOT

# 通用（只起 policy server）
SERVER_ARGS="--env DROID" docker compose -f scripts/docker/compose.yml up --build

# LIBERO 评测（server + 仿真一起起）
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# simple_client 冒烟（server + client 一起起）
SERVER_ARGS="--env DROID" docker compose -f examples/simple_client/compose.yml up --build
```

`SERVER_ARGS` 会透传给容器内的 `uv run scripts/serve_policy.py $SERVER_ARGS`。
缓存目录通过 `OPENPI_DATA_HOME` 挂载到容器内 `/openpi_assets`。

---

## 13. 关键参数与性能调优

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `num_steps` | `create_trained_policy(sample_kwargs={"num_steps": N})` | flow matching 采样步数，默认 10；调小更快、调大更准 |
| `action_horizon` | `Pi0Config` | 一次输出的动作步数（DROID=15，LIBERO=10，ALOHA=50） |
| `action_dim` | `Pi0Config` | pi05 固定 32；输出会被 `Outputs` 截取到机器人真实维度 |
| `max_token_len` | `Pi0Config` | pi05 默认 200（含 prompt + 离散化 state） |
| `dtype` | `Pi0Config` | 默认 `bfloat16` |
| `default_prompt` | `create_trained_policy` / `--default-prompt` | prompt 缺失时的兜底指令 |
| `pytorch_compile_mode` | `Pi0Config` | `max-autotune`（默认）/ `reduce-overhead` / `default` / `None` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 环境变量 | 训练时设为 `0.9` 让 JAX 用更多显存（推理一般不必） |

显存参考：推理 > 8 GB（RTX 4090 足够）。

---

## 14. 常见问题排查

| 现象 | 排查/解决 |
| --- | --- |
| `Config 'xxx' not found` | 用 `get_config` 会给出相近候选名；确认 `config.py` 的 `_CONFIGS` 里已注册 |
| `Asset id is required to load norm stats.` | 配置里 `asset_id` 为空；给 `AssetsConfig(asset_id=...)` 或设置 `repo_id` |
| checkpoint 下找不到 `norm_stats.json` | 本地微调产物必须包含 `assets/<asset_id>/norm_stats.json`；先跑 `scripts/compute_norm_stats.py` |
| `uv sync` 依赖冲突 | `rm -rf .venv` 后重跑；并 `uv self update` |
| 训练显存不足 | 设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`；或用 `--fsdp-devices <n>`；或关掉 EMA |
| 连接不上 policy server | 确认服务在跑、IP/端口正确、防火墙放行；从机器人端 `ping <server_ip>` 验证 |
| 推理慢 / 时延抖动 | DROID 每 chunk 0.5–1 s 属正常；机器人端尽量用有线网络；客户端提前 resize 图片 |
| PyTorch 报 `transformers_replace is not installed correctly` | 重做第 11.1 节的 `cp -r` 补丁，并确认 `transformers==4.53.2` |
| 动作维度不匹配 | 检查 `Inputs`/`Outputs` 中截取的维度是否与机器人 action space 一致 |
| CUDA / GPU 报错 | 不需要系统级 CUDA；若已装，考虑卸载以避免与 uv 装的库冲突；检查 nvidia-container-toolkit |
| 训练 loss 发散 | 查 `norm_stats.json` 里的 `q01/q99/std`，个别不常用维度的值过小会导致归一化后数值爆炸，可手动修正 |

---

## 附：一条命令跑通（本机已缓存 pi05_droid）

```bash
cd /home/go2/StreamingVLA/openpi
uv run scripts/serve_policy.py --env=DROID          # 终端 1
uv run examples/simple_client/main.py --env DROID   # 终端 2，应打印时延统计表
```

或直接 Python API：

```bash
cd /home/go2/StreamingVLA/openpi
uv run python -c "
import numpy as np
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

cfg = _config.get_config('pi05_droid')
ckpt = download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid')
p = policy_config.create_trained_policy(cfg, ckpt)
obs = {
    'observation/exterior_image_1_left': np.random.randint(256, size=(224,224,3), dtype=np.uint8),
    'observation/wrist_image_left': np.random.randint(256, size=(224,224,3), dtype=np.uint8),
    'observation/joint_position': np.random.rand(7),
    'observation/gripper_position': np.random.rand(1),
    'prompt': 'pick up the fork',
}
out = p.infer(obs)
print('action chunk:', out['actions'].shape, '| infer_ms:', out['policy_timing']['infer_ms'])
"
```
