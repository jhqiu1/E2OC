# E2OC: 大语言模型驱动的多目标组合优化算子协同演化

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/badge/Release-v1.0.0-green.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2601.17899-b31b1b.svg)](https://arxiv.org/abs/2601.17899)
[![Website](https://img.shields.io/badge/Website-E2OC-8A2BE2.svg)](https://jhqiu1.github.io/e2oc)
[**English**](README.md)

**E2OC** 是首个面向 MOEA 的多算子协同设计框架。它利用马尔可夫决策过程 (MDP) 分析算子间的耦合关系，通过蒙特卡洛树搜索实现算子设计策略与可执行代码的联合演化。现有基于 LLM 的自动启发式设计方法独立优化每个算子，忽略了一个根本性挑战：修改一个算子会重塑其他算子的搜索环境。E2OC 通过协同演化相互依赖的算子来解决这一问题，其模块化架构支持灵活扩展新组件。

**网站**: [https://jhqiu1.github.io/e2oc](https://jhqiu1.github.io/e2oc)

---

## 简介

### 设计范式

![Pipeline Comparison](figures/figure_Com_pipeline_01.png)

多目标进化算法 (MOEA) 依赖邻域搜索算子（交叉、变异、局部搜索），其设计对性能至关重要。三种范式对比如下：

1. **专家设计** &mdash; 成本高昂，领域特定，难以泛化
2. **单算子 AHD** (EoH, FunSearch, ReEvo 等) &mdash; 独立优化每个算子，忽略耦合效应
3. **E2OC (我们提出的方法)** &mdash; 首个通过 MDP 分析和 MCTS 搜索协同演化相互依赖算子的框架

### E2OC 框架

![E2OC Framework](figures/figure_framwork_01.png)

E2OC 包含四个核心组件：

**热启动初始化。** 使用算法生成器 $\mathcal{G}$ 对每个算子进行独立演化，构建结构化的**设计思想空间**，从精英算子中提取语义级的改进建议。这为每个算子预先构建了一组高质量的设计思想集合，将搜索空间从无界约束为可处理的。

**多领域设计思想语言空间。** 算子设计思想对应于语义级表示，每个思想定义了一个决策范式或改进方向。内部关系刻画了同一算子不同思想的相对优劣和继承关系，外部关系表示不同算子间的跨域依赖（互补、冲突或互斥）。MDP 用于分析这些耦合关系。

**渐进式 MCTS 搜索。** 探索跨算子设计思想的**组合**，识别有前景的设计策略。设计空间被建模为一棵树，节点状态代表不同领域的设计思想。基于 UCB 的节点选择平衡探索与利用。有界空间反而通过集中评估预算提升了性能。

**算子轮换演化。** 在选定的设计策略下，每个算子在真实多算子系统的**上下文中**被演化和评估。轮换过程中通过逐一替换算子并评估其对整体性能的影响来逐步更新。**算法生成器可插拔**，基于 [LLM4AD](https://github.com/Optima-CityU/LLM4AD) 平台构建，支持多种方法，包括 **EoH、FunSearch、ReEvo 和 MCTS-AHD**。

### 设计哲学

| 理念 | 说明 |
|------|------|
| **语义级搜索优于代码级搜索** | 在设计意图空间而非语法变异空间中搜索 |
| **有界优于无界** | 筛选后的思想集合提升采样效率 |
| **协同设计产生互补性** | 演化算子自发形成功能分工 |

### 关键结果

| 基准测试 | vs. 专家设计 | vs. 最佳 AHD (单算子) | vs. 最佳 AHD (多算子) |
|-----------|-----------|----------------------|---------------------|
| Bi-FJSP + NSGA-II | **+22.00% HV** | +7.3% | +12.2% |
| Bi-TSP + NSGA-II | **+14.00% HV** | | |
| Tri-FJSP + NSGA-II | **+17.36% HV** | | |

**成本**: 每个设计任务约 $1.14 (DeepSeek-Chat)。**泛化能力**: TSP-100 上训练的算子可迁移至 TSP-200，HV 提升 +22.06%。

![Convergence Analysis](figures/figure_convergence_ana_01.png)

E2OC 能够**持续优化**：Round 1 &rarr; Round 2 (+0.8% HV) &rarr; Round 3 (+1.6% HV)，不会陷入收敛陷阱。算子可在不同问题规模间泛化 (TSP-100 &rarr; TSP-150/200)。

---

## 环境要求

- Python >= 3.9
- `numpy < 2.0.0`, `scipy`, `torch`, `networkx`
- `requests`, `openai` (用于 LLM API 调用)
- `hvwfg >= 1.0.0` (用于超体积计算)
- 可选: `tensorboard`, `wandb`, `matplotlib`, `codebleu`

MOEA 引擎为纯 Python 实现，已包含在仓库中，无需额外安装外部 MOEA 框架。

---

## 快速开始

### 安装

```bash
cd E2OC

# 方式 1: 从 requirements.txt 安装
pip install -r requirements.txt

# 方式 2: 以可编辑模式安装包（推荐用于开发）
pip install -e .
```

项目使用 `pyproject.toml` 作为现代化的 Python 构建配置。

### 步骤 1: 配置 LLM API

编辑 [`examples/tsp_biobjective/config.py`](examples/tsp_biobjective/config.py)：

```python
LLM_CONFIG = {
    "host": "api.deepseek.com",      # LLM API 主机
    "api_key": "your_api_key_here",  # LLM API 密钥
    "model": "deepseek-chat",         # LLM 模型
    "timeout": 120,
}
```

### 步骤 2: 运行 E2OC

```bash
python run_e2oc.py
```

框架将执行四组件协同演化流程：
1. **热启动**: 通过独立演化初始化算子种群
2. **语言空间**: 构建多领域设计思想空间
3. **渐进式 MCTS 搜索**: 探索设计策略组合
4. **算子轮换演化**: 在上下文中协同演化算子

结果保存到 `outputs_tsp/`。

### 步骤 3: 查看结果

| 文件 | 内容 |
|------|------|
| `storages.json` | 算子性能历史 |
| `best_results_*.json` | 找到的最佳算子 |
| `mcts_history.txt` | MCTS 搜索历史 |
| `convergence_curve.jpg` | 收敛曲线 |

---

## 在你的应用中使用 E2OC

*即将推出:* 将 E2OC 适配到新多目标优化问题的详细指南，以及独立使用各生成器（EoH, FunSearch 等）的教程。

---

## 引用

如果你发现 E2OC 对你的研究或应用项目有帮助，请引用：

```bibtex
@inproceedings{qiu2026e2oc,
  title={Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization},
  author={Qiu, Junhao and Chen, Xin and Ge, Liang and Lin, Liyong and Lu, Zhichao and Zhang, Qingfu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  url={https://arxiv.org/abs/2601.17899}
}
```

如果你对 LLM4Opt 或 E2OC 感兴趣，可以：

- 通过邮箱联系我们: junhaoqiu2-c@my.cityu.edu.hk
- 访问 LLM4Opt 参考文献和研究论文集合
- 加入我们的团队 (即将推出)

如果您在使用代码时遇到任何困难，请通过上述方式联系我们或提交 issue。

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
