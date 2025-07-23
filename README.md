# Unsloth-Factory 🦥  
**在单张16G显卡上完成7B/8B大模型“SFT微调 → 评估 → 合并 → 量化 → Ollama部署”全流程的极速框架**

> 基于 [Unsloth](https://github.com/unslothai/unsloth) 的高性能微调套件，专为小显存场景深度优化，训练速度提升 **2-3 倍**，显存占用降低 **50-70%**。  

---

## 🚀 1 分钟速览

| 特性         | 说明                                                       |
| ------------ | ---------------------------------------------------------- |
| **模型**     | Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Qwen2-7B-Instruct … |
| **最低显存** | 16 GB（Tesla T4 实测）                                     |
| **量化**     | 原生 FP16 → GGUF Q4_0/Q6_K，体积压缩 **75 %**              |
| **全流程**   | SFT / DPO / 验证 / 评估 / 对话 / 合并 / 量化 / Ollama 部署 |
| **领域知识** | 医疗、法律、心理学、考试 4 大内置知识库                    |
| **CLI**      | 1 条命令完成端到端训练与部署                               |

---

## 📦 环境准备

```bash
# 1. 创建环境
conda env remove -n unsloth -y
conda create -n unsloth python=3.11 -y
conda activate unsloth

# 2. 安装 PyTorch（清华源）
pip install torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 Unsloth 及其他依赖
pip install unsloth==2025.6.5 unsloth-zoo==2025.6.4 \
  accelerate==1.7.0 xformers==0.0.29.post3 triton==3.2.0 \
  transformers==4.52.4 peft==0.15.2 trl==0.12.0 datasets==3.6.0 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install evaluate gputil
```

## 项目结构

unsloth-factory/
├── cli.py                       # 统一命令行入口
├── src/
│   ├── core/                    # 模型 & 数据集工厂
│   ├── trainers/                # SFT / DPO 训练器
│   ├── evaluators/              # C-Eval 等评测
│   ├── validators/              # 领域验证
│   ├── chat/                    # 流式对话
│   ├── knowledge/               # 医疗|法律|心理学|考试知识库
│   └── utils/
├── datasets/                    # 训练数据
└── output/                      # 模型/日志/量化产物

## 5 步走：从数据到部署

> 以下用 **医疗领域 Qwen3-8B** 为例，数据集：`medical_o1_alpaca.json`

### ① SFT 微调

python cli.py sft \
  --model /path/to/Qwen3-8B \
  --domain medical \
  --dataset datasets/medical_o1_alpaca.json \
  --dataset_format alpaca \
  --output_dir output/sft-qwen3-8b \
  --epochs 3 \
  --max_seq_length 4096 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --learning_rate 1e-5 \
  --save_steps 200 \
  --resume auto

*10h 完成 6 000 条样本训练（Tesla T4）*

### ② 领域验证

python cli.py validate \
  --model /path/to/Qwen3-8B \
  --adapter output/sft-qwen3-8b/final_adapter \
  --dataset datasets/medical_o1_alpaca.json \
  --max_samples 10 \
  --output_dir output/sft-qwen3-8b/validation_results

### ③ 基准测试（C-Eval）

python cli.py evaluate \
  --task ceval \
  --model /path/to/Qwen3-8B \
  --adapter output/sft-qwen3-8b/final_adapter \
  --task_dir datasets/ceval-exam \
  --n_shot 10 \
  --save_dir output/sft-qwen3-8b/evaluation_results

### ④ 权重合并 & 量化

合并 LoRA → 完整模型

python cli.py merge \
  --model /path/to/Qwen3-8B \
  --adapter output/sft-qwen3-8b/final_adapter \
  --output output/sft-qwen3-8b/merged_model

转换为 GGUF

python /path/to/llama.cpp/convert_hf_to_gguf.py \
  --outfile qwen3-8b.gguf ./merged_model

量化 Q4_0（4-bit）

llama-quantize qwen3-8b.gguf qwen3-8b-q4.gguf q4_0

### ⑤ Ollama 部署

生成 Modelfile（注意修改 FROM 路径）

ollama show qwen3:8b --modelfile > Modelfile

编辑 Modelfile 中 FROM 指向 qwen3-8b-q4.gguf 绝对路径

ollama create qwen3-8b-q4 -f Modelfile
ollama run qwen3-8b-q4:latest

## 🔍 FAQ

| 问题                            | 解答                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| **显存不足？**                  | 启用 `--gradient_checkpointing`、`--max_seq_length 2048`、`--batch_size 1` |
| **训练中断？**                  | 使用 `--resume auto` 自动断点续训                            |
| **量化后质量下降？**            | 改为 `q6_K` 量化或复杂场景回退 FP16                          |
| **DeepSeek-R1-0528 无法量化？** | llama.cpp 尚未支持，等待后续版本                             |

## 性能对比（Qwen3-8B）

| 场景          | 原生 FP16 | Q4\_0 量化 | 备注               |
| ------------- | --------- | ---------- | ------------------ |
| 模型大小      | 15.6 GB   | 4.5 GB     | -                  |
| 推理速度      | 11 tok/s  | 28 tok/s   | RTX 4090           |
| C-Eval 平均分 | 22.6 %    | 22.6 %     | 医疗微调前后持平\* |
| 医疗问答 BLEU | 8.5       | 6.2        | 量化后略有下降     |