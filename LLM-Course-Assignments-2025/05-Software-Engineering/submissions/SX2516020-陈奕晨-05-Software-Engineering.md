# 大模型实验报告

## 整体思路

Code generation任务上，想通过改变训练数据的学习模式，通过逐步给模型数据学习，实时获取运行结果，分配下一轮训练的任务，包括新题目和需要回顾的旧题，进行**lora微调**；同时使用过程中每个问题的多个执行结果，构建DPO的pair对，再进行**DPO**

## 实验设置

Qwen2.5-Coder-1.5B/7B

Deepseek-Coder-6.7B

Deepseek-Coder-6.7B-Instruct

推理使用vllm0.11.1

## 算力资源

Qwen2.5-Coder-1.5B在2*Nvidia L40s上进行

Qwen2.5-Coder-7B，Deepseek-Coder-6.7B/Instruct在华为云ModelArts上进行，使用4*Ascend910B(64G)进行

## 代码实现

### 