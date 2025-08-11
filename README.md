# Unsloth-Factory 🦥  
**在单张16G显卡上完成7B/8B/14B大模型“SFT微调 → 评估 → 合并 → 量化 → Ollama部署”全流程的极速框架**

> 基于 [Unsloth](https://github.com/unslothai/unsloth) 的高性能微调套件，专为小显存场景深度优化，训练速度提升 **2-3 倍**，显存占用降低 **50-70%**。  最新版已经能微调qwen3-14B模型，达到了tesla T4显卡的极限。

---

## 🚀 1 分钟速览

| 特性         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **模型**     | Qwen3-14B、Qwen3-8B、Qwen2.5-7B、DeepSeek-R1-Qwen-7B、Qwen2-7B-Instruct |
| **最低显存** | 16 GB（Tesla T4 实测）                                       |
| **量化**     | 原生 FP16 → GGUF Q4_0/Q6_K，体积压缩 **75 % /66%**           |
| **全流程**   | SFT / DPO / 验证 / 评估 / 对话 / 合并 / 量化 / Ollama 部署   |
| **领域知识** | 医疗、法律、心理学、考试 4 大内置知识库                      |
| **CLI**      | 1 条命令完成端到端训练与部署                                 |

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



## 各个大模型微调结果对比

通过unsloth-factory微调了4个模型（deepseek-r1-qwen3-8b不能导出gguf格式文件），对比了DeepSeek-R1-Distill-Qwen-7B，Qwen2-7B-Instruct，Qwen3-8B，Qwen3-14B，Qwen3-14B量化版本（q6_k,部署在ollama上），以及Qwen3-14B原始版本，分别对6个模型提出了3个相同问题：

1. 问题1：一名70岁的男性患者因胸痛伴呕吐16小时就医，心电图显示下壁导联和右胸导联ST段抬高0.1~0.3mV，经补液后血压降至80/60mmHg，患者出现呼吸困难和不能平卧的症状，体检发现双肺有大量水泡音。在这种情况下，最恰当的药物处理是什么？
2. 问题2：对于一名60岁男性患者，出现右侧胸疼并在X线检查中显示右侧肋膈角消失，诊断为肺结核伴右侧胸腔积液，请问哪一项实验室检查对了解胸水的性质更有帮助？
3. 问题3：一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？

 考虑到篇幅，这里只贴结论

**一、医学问答模型质量对比和排序**

1. qwen3-14-微调版本 - 在所有三个问题上都提供了专业、详细、准确的回答
2. qwen3-14-原始版本 - 与微调版本质量接近，但在部分细节上略逊
3. Qwen3-8B-微调版本 - 专业性高，但某些方面不如qwen3-14系列深入
4. Qwen2-7B-Instruct-微调版本 - 专业性较高，但分析不够全面深入
5. qwen3-14-量化q6K版本 - 专业性一般，缺乏深度和准确性
6. DeepSeek-R1-Qwen-7B-微调版本 - 专业性差，问题1甚至出现严重错误诊断

  

**二、各问题详细质量对比**

**问题1：急性心肌梗死患者的药物处理**

最佳回答：qwen3-14-微调版本 & qwen3-14-原始版本

优势：

- 准确诊断为"急性下壁和右室心肌梗死"并识别"右心室功能不全和低血压性休克"
- 详细解释了心电图特征与临床表现的关联
- 提出关键处理原则：避免过度补液（针对右室梗死的特殊处理）
- 明确推荐去甲肾上腺素为首选升压药，而非多巴胺
- 指出应避免使用硝酸甘油（右室梗死时可能加重病情）
- 提供了清晰的药物处理优先顺序和剂量建议
- 包含再灌注治疗建议和机械支持考虑

其他模型问题：

- qwen3-14-量化q6K：仅推荐呋塞米，忽略了升压药物，未认识到右室梗死的特殊性
- DeepSeek-R1-Qwen-7B：错误诊断为"急性肺栓塞"，建议使用抗组胺药物（完全错误）
- Qwen3-8B-微调：虽专业但未强调右室梗死的特殊处理细节
- Qwen2-7B-Instruct：处理方案较全面，但缺乏对右室梗死特殊性的深入分析

 

**问题2：肺结核伴胸腔积液的实验室检查**

最佳回答：Qwen3-8B-微调版本 & qwen3-14-微调版本

优势：

- Qwen3-8B-微调：明确指出胸水腺苷脱氨酶（ADA）检测是关键检查
- 详细解释ADA>45 U/L对结核性积液的诊断价值（特异性>90%）
- 分析了其他检查的局限性（如细胞学检查阳性率低）
- 提供了明确的临床决策路径
- qwen3-14-微调：全面分析了胸水检查
- 区分了渗出性/漏出性胸水的判断标准
- 提及ADA检测、葡萄糖水平和pH值对结核性胸膜炎的诊断价值

其他模型问题：

- qwen3-14-量化q6K：仅强调细胞学分析，未突出ADA的特殊价值
- DeepSeek-R1-Qwen-7B：混淆了多种检查，未明确指出关键检查
- Qwen2-7B-Instruct：虽提及ADA，但未将其作为核心检查重点强调

 

**问题3：儿童头皮疾病的中医诊断**

最佳回答：qwen3-14-微调版本 & Qwen3-8B-微调版本

优势：

qwen3-14-微调：

- 诊断为"湿热毒蕴"所致的"痈"或"疔疮"
- 详细分析病因病机（湿热邪气侵袭，夏季湿热较重）
- 提供了中医辨证分型和具体治疗原则
- 包含西医排查建议和注意事项

Qwen3-8B-微调：

- 诊断为"岩"或"湿毒郁结"之证，属于"恶疮"或"头疮"范畴
- 详细分析中医病名辨证和辨证要点
- 提供了系统的治疗原则、注意事项和鉴别诊断

其他模型问题：

- qwen3-14-原始版本：诊断为"发际疮"，分析较全面但不如微调版本深入
- qwen3-14-量化q6K：诊断为"湿热疮"或"血热疮"，分析较为简单
- DeepSeek-R1-Qwen-7B：使用非标准中医术语
- Qwen2-7B-Instruct：仅诊断为"疖"或"痈"，分析不够深入

  

**三、差异原因分析**

1. 模型规模与架构差异

- qwen3-14B（140亿参数） 比 Qwen3-8B（80亿参数）和 Qwen2-7B（70亿参数）具有更强的知识表示能力
- 更大的模型能更好地捕捉医学专业知识的复杂模式和关联

2. 微调数据质量与针对性

- qwen3-14-微调版本 使用了3000条专业医学术语进行微调，特别针对医学领域进行了优化
- 量化版本（qwen3-14-量化q6K）在量化过程中损失了部分精度，影响了专业判断
- DeepSeek-R1-Qwen-7B 虽然也是7B级别，但微调数据可能不够专业或针对性不强

3. 医学专业知识深度

qwen3-14系列 展示出对医学概念的准确理解：

- 正确识别右室梗死需要避免过度补液的特殊处理
- 理解ADA检测对结核性胸膜炎的特异性诊断价值
- 能将症状、体征与病理生理机制准确关联

低质量模型：

- DeepSeek-R1-Qwen-7B在问题1中错误诊断为急性肺栓塞
- qwen3-14-量化q6K在问题1中忽略了右室梗死的特殊处理原则

4. 临床推理能力

高质量模型（qwen3-14系列）展示出更强的临床推理能力：

- 能理解为什么右室梗死患者补液后血压下降
- 能将多个临床表现整合为一致的诊断思路
- 能提供符合当前指南的治疗建议

低质量模型往往机械应用一般原则，缺乏对特殊情况的针对性处理

5. 回答结构与专业性

高质量模型的回答结构清晰：

- 明确诊断 → 详细分析 → 具体处理建议 → 总结
- 使用专业术语准确，逻辑严密

低质量模型的回答往往：

- 信息杂乱，缺乏逻辑结构
- 混淆概念（如DeepSeek-R1将心梗误诊为肺栓塞）
- 提供不恰当的治疗建议（如建议使用抗组胺药物）

 

 **四、总结建议**

模型质量差异主要源于模型规模、微调数据质量与针对性、医学专业知识深度三个关键因素。qwen3-14-微调版本在所有方面都表现出色，证明了高质量专业数据微调对特定领域性能提升的重要性。

在医学领域应用中，专业领域的高质量微调比单纯增加模型参数更为关键。即使是较小的模型（如Qwen3-8B），通过针对性的专业微调，也能在特定问题上提供非常专业的回答。

对于需要高精度医学建议的应用场景，应优先选择经过专业医学数据微调的大模型，而非仅依赖模型参数规模。

-  临床场景：优先选择Qwen3-14B微调版，其诊断、药物、检查建议最接近真实医疗决策。
- 资源受限：Qwen3-8B微调可作为次选，但需人工复核。
- 避免使用：DeepSeek-7B与量化Q6K在医学准确性上存在显著缺陷，可能误导临床判断。
- 差异核心：模型规模（参数量） + 专业语料微调质量 + 量化损失共同决定回答的临床可靠性。

 

最终推荐：医学场景优先选用 Qwen3-14B微调版，避免使用7B以下模型或量化版本。若需部署轻量模型，Qwen3-8B微调版为最低可用底线。

 