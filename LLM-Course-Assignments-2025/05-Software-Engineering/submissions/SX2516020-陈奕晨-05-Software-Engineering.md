# 大模型实验报告：05-Software-Engineering

## 整体思路

Code generation任务上，想通过改变训练数据的学习模式，通过逐步给模型数据学习，实时获取运行结果，分配下一轮训练的任务，包括新题目和需要回顾的旧题，进行**lora微调**；同时使用过程中每个问题的多个执行结果，构建DPO的pair对，再进行**DPO**

## 实验设置

### 模型

Qwen2.5-Coder-1.5B/7B

Deepseek-Coder-6.7B

Deepseek-Coder-6.7B-Instruct

### 数据集

KodCode-V1：https://huggingface.co/datasets/KodCode/KodCode-V1

该数据集包含12个独立子集，覆盖多个领域（从算法知识到特定软件包知识）和难度级别（从基础编程练习到面试题及竞赛编程挑战）。KodCode 同时支持监督式微调（SFT）和强化学习调优（RL tuning）。

数据集包含：题目来源，题目描述，测试代码（需要使用指定的函数定义），标准解法等等

### 推理

使用vllm0.11.1

## 算力资源

Qwen2.5-Coder-1.5B在2*Nvidia L40s上进行

Qwen2.5-Coder-7B，Deepseek-Coder-6.7B/Instruct在华为云ModelArts上进行，使用4*Ascend910B(64G)进行

## 核心代码实现

### generator

可以根据给出的prompt，让模型生成代码

```py
def generate(self, prompts, num_candidates=1):
        
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=['<|endofblock|>', '<|endofmessage|>']
        )
        
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_candidates)
        if self.lora_request:
            outputs = self.llm.generate(expanded_prompts, sampling_params, lora_request=self.lora_request)
        else:
            outputs = self.llm.generate(expanded_prompts, sampling_params)
            
        completions: List[List[str]] = [[] for _ in range(len(prompts))]
        
        for i, out in enumerate(outputs):
            prompt_index = i // num_candidates
            for o in out.outputs:
                completions[prompt_index].append(o.text)
            
        return completions
```

### prompt生成

可以根据不同模型，区别base/instruct模型进行构造prompt。由于KodCode数据集需要指定的函数定义，否则会无法使用数据集提供的测试代码进行测试，所以要在prompt中指定好函数定义

```py
def build_prompt_instruct(question, test_info):
    prompt = "<|im_start|>system\n"
    prompt += "You are a Python expert. Write only the function implementation without explanations.<|im_end|>\n"
    
    prompt += "<|im_start|>user\n"
    prompt += f"{question}\n\n"
    prompt += "Complete this function:\n"
    prompt += f"```python\n{test_info['function_declaration']}\n    pass\n```<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_deepseek(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "### Instruction:\n"
    prompt += "Write a complete Python function for the problem with the provided function declaration. "
    prompt += "Output only clean Python code without any comments, docstrings, or explanations.\n\n"
    prompt += "Question: " + question.strip() + "\n"
    prompt += "Function declaration: " + function_declaration + "\n\n"
    prompt += "### Response:\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_codellama(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "[INST] <<SYS>>\n"
    prompt += "You are an expert Python programmer. "
    prompt += "You always write clean, efficient, and correct code. "
    prompt += "You output only code without any explanations or comments.\n"
    prompt += "<</SYS>>\n\n"
    
    prompt += "Write a complete Python function for the problem with the provided function declaration.\n\n"
    prompt += f"Question: {question.strip()}\n"
    prompt += f"Function declaration: {function_declaration}\n"
    prompt += "[/INST] ```python\n"
    
    return prompt
    

def build_prompt(question, test_info, model_type=None):
    if model_type == "Deepseek":
        return build_prompt_deepseek(question, test_info)
    elif model_type == "Qwen":
        pass
    elif model_type == "CodeLlama":
        return build_prompt_codellama(question, test_info)
    else:
        function_declaration = test_info['function_declaration']
        prompt = "Write a complete Python function for the problem with the provided function declaration. Output only clean Python code without any comments, docstrings, or explanations.\n"
        prompt += "Question: " + question.strip() + "\n"
        prompt += "Function declaration: " + function_declaration + "\n\n"
        prompt += "```python\n"
    
    return prompt
```

