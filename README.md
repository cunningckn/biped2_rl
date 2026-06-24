# Biped2 RL — 双足机器人强化学习训练

基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 与 [Isaac Gym](https://developer.nvidia.com/isaac-gym)，在平面地形上训练 **biped2** 双足机器人的行走策略。

**仓库**: https://github.com/cunningckn/biped2_rl

---

## 机器人概览

| 项目 | 说明 |
|------|------|
| 模型 | `resources/robots/biped2/urdf/biped2.urdf` |
| 自由度 | 10（每腿 5 关节） |
| 关节链 | HipYaw → HipRoll → HipPitch → KneePitch → AnklePitch |
| 初始高度 | 0.45 m |
| 控制方式 | 位置 PD（`action_scale = 0.25`） |

```
        base_link
       /         \
  Right leg    Left leg
  (5 DOF)      (5 DOF)
```

---

## 训练任务

| Task 名称 | 环境类 | 说明 |
|-----------|--------|------|
| `biped2_flat` | `Biped2` | 基础平面行走：速度跟踪 + 单腿支撑 + 低速站直 |
| `biped2_gait_flat` | `Biped2Gait` | 在基础任务上增加**相位步态奖励**，约束摆动/支撑相足部力与速度 |

日志目录：

- `logs/flat_biped2/` — 基础行走
- `logs/flat_biped2_gait/` — 步态约束训练

---

## 奖励设计

### 通用（`biped2_flat`）

| 奖励 | 作用 |
|------|------|
| `tracking_lin_vel` / `tracking_ang_vel` | 跟踪线速度 / 偏航角速度指令 |
| `feet_air_time` | 单腿支撑时奖励迈步持续时间（`min` 设计，鼓励交替步态） |
| `low_speed_upright` | 低速时跟踪默认关节姿态，保持站直 |
| `no_fly` | 运动时禁止双脚同时离地 |
| `termination` | 躯干 / 髋部异常接触终止 |

### 步态相位（`biped2_gait_flat` 额外）

基于 `gait_clock` 生成摆动相 / 支撑相掩码，仅在 `||cmd_xy|| > 0.1` 时生效：

| 奖励 | 相位 | 目标 |
|------|------|------|
| `gait_feet_frc_perio` | 摆动相 | 足底力 ≈ 0 |
| `gait_feet_spd_perio` | 支撑相 | 足底速度 ≈ 0（防滑动） |
| `gait_feet_frc_support_perio` | 支撑相 | 足底有足够支撑力 |

步态参数见 `legged_gym/envs/biped2/biped2_config.py` 中 `Biped2GaitFlatCfg.rewards.gait`。

---

## 安装

依赖与上游 legged_gym 相同，需提前安装 **Isaac Gym Preview 3** 与 **rsl_rl**。

1. Python 3.8 虚拟环境
2. PyTorch（与 CUDA 版本匹配，例如 cu113）：
   ```bash
   pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
     -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
3. Isaac Gym：
   ```bash
   cd isaacgym/python && pip install -e .
   ```
4. [rsl_rl](https://github.com/leggedrobotics/rsl_rl) v1.0.2：
   ```bash
   git clone https://github.com/leggedrobotics/rsl_rl
   cd rsl_rl && git checkout v1.0.2 && pip install -e .
   ```
5. 本仓库：
   ```bash
   cd biped2_rl && pip install -e .
   ```

---

## 使用方法

### 训练

```bash
cd biped2_rl

# 基础平面行走
python legged_gym/scripts/train.py --task=biped2_flat

# 步态相位约束训练
python legged_gym/scripts/train.py --task=biped2_gait_flat
```

常用参数：

```bash
# 无渲染（更快）
python legged_gym/scripts/train.py --task=biped2_gait_flat --headless

# 恢复训练
python legged_gym/scripts/train.py --task=biped2_gait_flat --resume

# 指定环境数量
python legged_gym/scripts/train.py --task=biped2_flat --num_envs=2048
```

训练开始后按 `v` 可关闭实时渲染以提升速度。策略保存在：

```
logs/<experiment_name>/<date>_<run_name>/model_<iteration>.pt
```

### 播放策略

```bash
python legged_gym/scripts/play.py --task=biped2_gait_flat
```

在 `play.py` 顶部修改 `CMD_X` / `CMD_Y` / `CMD_YAW` 可固定速度指令进行测试。

### TensorBoard

```bash
tensorboard --logdir logs/flat_biped2_gait
```

---

## 代码结构（biped2 相关）

```
legged_gym/
├── envs/biped2/
│   ├── biped2.py          # Biped2 / Biped2Gait 环境及奖励实现
│   └── biped2_config.py   # 环境参数、奖励权重、步态配置
├── envs/__init__.py       # 注册 biped2_flat / biped2_gait_flat
├── scripts/
│   ├── train.py
│   └── play.py
└── resources/robots/biped2/
    ├── urdf/biped2.urdf
    └── meshes/
```

---

## 调参建议

| 现象 | 可尝试 |
|------|--------|
| 不迈步 / 交替差 | 增大 `feet_air_time` 权重 |
| 低速时姿态塌 | 增大 `low_speed_upright` |
| 步态周期不对 | 调整 `gait.cycle`、`air_ratio_l/r` |
| 支撑脚滑动 | 增大 `gait_feet_spd_perio` |
| 摆腿拖地 | 增大 `gait_feet_frc_perio` |

奖励权重均在 `biped2_config.py` 的 `rewards.scales` 中，设为 `0` 可关闭对应项。

---

## 致谢

本项目基于以下开源工作：

- [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH Zurich / NVIDIA
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — PPO 实现
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA

原始论文：[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)

---

## License

BSD-3-Clause（与上游 legged_gym 一致）
