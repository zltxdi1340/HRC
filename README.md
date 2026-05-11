# HRC: Causal-Graph-Guided Hierarchical Reinforcement Learning

本项目研究 MiniGrid / BabyAI 环境中的层次强化学习迁移问题。核心目标是构建可复用的策略模块库，使智能体能够在多个相关任务之间高效迁移。

当前重点方法为：

- 使用 G2 动作模板级模块构建可迁移策略模块；
- 使用 B Sequential Aggregation 顺序聚合方法构建模块库；
- 在新任务中通过模块组合实现 zero-shot 或少量适配迁移。

## 1. Project Overview

本项目目前包含三类模块粒度：

| 粒度 | 含义 | 示例 |
|---|---|---|
| G0 | 粗粒度状态谓词模块 | `has_key`, `opened_door`, `has_target` |
| G1 | 因果边模块 | `near_key -> has_key`, `has_key -> opened_door` |
| G2 | 动作模板模块 | `approach_key`, `pickup_key`, `approach_door`, `toggle_door` |

当前实验表明，在门链迁移任务中，G2 动作模板模块在成功率、训练成本和执行步数上表现最优。

## 2. Repository Structure

```text
HRC_granularity/
├── env/                         # 状态节点、子目标检测、节点别名
├── rl/                          # PPO 训练环境、G2 模板环境、模型和 callback
├── configs/                     # 任务链配置、实验配置、模块映射配置
├── docs/                        # 项目结构、实验记录、模块库说明
├── module_library/              # 可复用模块库，本地保存模型，GitHub 只保存 metadata
├── results_*/                   # 实验结果 CSV
├── runs_*/                      # 训练日志与模型，本地保留，GitHub 忽略
├── hrc_stage1.py                # 因果图构建 / Stage 1 探索
├── causal_discovery_simple.py    # 简化因果发现算法
├── train_sb3_g2_ppo_multitask.py # G2 多任务训练脚本
├── train_sb3_gc_ppo_multitask.py # G0/G1 多任务训练脚本
└── eval_*.py                    # 各类评估脚本