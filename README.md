## 微调框架优化：支持Qwen3-14b  

这是一个专注于在 Tesla T4 16GB GPU 上运行的微调框架。框架提供从数据准备、模型训练（SFT/DPO）、合并、量化、评估、验证到最终部署（REST API + Web/CLI 客户端）的全流程支持。核心设计思想是模块化、可扩展和实验可管理，通过统一的命令行接口（cli.py）集成各功能模块。

---

## 🚀 1 分钟速览

**1、设计要点**

1. 模块划分清晰，职责单一：每个模块独立负责特定功能，便于维护和扩展。
2. 实验管理：自动记录每次训练的配置、检查点、结果，支持恢复和对比，是生产环境必不可少的部分。
3. 针对tesla T4的优化策略：
   - 4-bit量化加载（NF4）
   - 限制并发数（1-2）
   - 生成后清理KV缓存
   - 使用CPU卸载加载大模型（14B）
4. 多领域支持：通过知识库动态注入领域知识，验证时可进行针对性检查，未来可扩展更多领域。
5. 量化流程完整：支持GGUF（用于llama.cpp）和GPTQ/AWQ（用于vLLM等），便于不同部署场景。
6. 验证体系完善：不仅计算指标，还评估思维链质量和多维度评分，确保模型输出安全合理。
7. API兼容OpenAI：降低集成成本，流式输出格式标准，客户端可复用现有库。
8. Web UI与CLI分离：CLI适合调试，Web UI提供可视化界面，均通过Rest-API调用。

---

## 📦 环境准备

```bash
这里需要设定2个环境，因为支持gptq/awq格式需要llmcompressor，有潜在冲突风险

---uf8---
conda remove --name uf8 --all
conda create -n uf8 python=3.11 -y
conda activate uf8

pip install \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --trusted-host download.pytorch.org

# 安装Unsloth和相关依赖
pip install \
  unsloth==2025.10.6 \
  unsloth-zoo==2025.10.6 \
  accelerate==1.7.0 \
  xformers==0.0.29.post3 \
  triton==3.2.0 \
  transformers==4.52.4 \
  peft==0.15.2 \
  trl==0.12.0 \
  datasets==3.6.0 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn

pip install evaluate
pip install gputil


ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
unsloth-zoo 2025.10.6 requires transformers!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,<=4.56.2,>=4.51.3, but you have transformers 4.57.6 which is incompatible.
unsloth 2025.10.6 requires transformers!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,<=4.56.2,>=4.51.3, but you have transformers 4.57.6 which is incompatible.
torchvision 0.21.0+cu124 requires torch==2.6.0, but you have torch 2.10.0 which is incompatible.
xformers 0.0.29.post3 requires torch==2.6.0, but you have torch 2.10.0 which is incompatible.
torchaudio 2.6.0+cu124 requires torch==2.6.0, but you have torch 2.10.0 which is incompatible.

这里在尝试安装llmcompressor会让torch库升级，为了顺利完成 GPTQ 量化，建议创建一个新的干净 Conda 环境，所以这里创建vllm环境，生成的权重文件直接在vllm下运行
且 llmcompressor 虽然安装了，但导入失败（可能由于 torch 版本过高）。

---vllm---
conda remove --name vllm --all
conda create -n vllm python=3.11 -y
conda activate vllm

# 1.2 如果使用pip，可以按以下方式安装特定CUDA版本的vLLM：
# 安装CUDA 12.8版本的vLLM pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
# 安装CUDA 11.8版本的vLLM pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# 安装CUDA 12.6版本的vLLM （当前使用这个版本）
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install llmcompressor

sudo lsof -i :8000 
vllm 默认是 8000 端口

```

## 架构

![](.\img\arch.png)



## 代码说明

**1、微调代码的技术特点：**

1. 支持多种训练方法：SFT（监督微调）、DPO（直接偏好优化）
2. 自动实验管理：创建实验ID、保存配置、检查点恢复
3. 多领域支持：医学、金融、法律、教育、心理等，通过知识库增强响应
4. 模型量化：GGUF（通过llama.cpp）、GPTQ/AWQ（通过llm-compressor）
5. 验证与评估：思维链质量、多维度评分、C-Eval评测
6. 部署服务：OpenAI兼容API，流式输出，并发控制（T4优化）
7. 客户端：命令行交互（带颜色区分）、Web UI（代理模式）

 

**2、平台代码说明**

**1）、根文件**

cli.py：主命令行入口，解析子命令，分发到各模块，包含实验管理集成。

 

**2）、src/trainers/ -** **训练模块**

base_trainer.py：训练器基类，初始化模型、tokenizer、数据集，应用LoRA，保存模型。

sft_trainer.py：SFT训练器，使用trl的SFTTrainer，封装训练参数和流程。

dpo_trainer.py：DPO训练器，使用trl的DPOTrainer。

trainer_factory.py：根据方法名创建对应训练器实例。

 

**3）、src/core/ -** **核心组件**

model_factory.py：模型加载工厂，支持Unsloth和标准Transformers，检测模型类型，合并适配器，获取LoRA目标模块。

dataset_factory.py：数据集加载工厂，支持多文件混合、数据限制、领域系统提示。

template_manager.py：模板管理器（当前未使用，可能用于未来对话模板管理）。

experiment_manager.py：实验管理器，创建实验、保存配置、管理检查点、恢复训练、列出/清理实验。

 

**4）、src/validators/ -** **验证模块**

cot_validator.py：思维链验证器，检测CoT格式，评估推理质量和一致性。

validator.py：模型验证器，加载模型生成响应，调用cot_validator，保存结果。

advanced_validator.py：高级多维度验证器，基于期望输出评分，使用知识库增强。

 

**5）、src/evaluators/ -** **评估模块**

ceval_evaluator.py：C-Eval评估实现，加载数据集、构建提示、提取答案、计算准确率。

evaluator.py：评估器主类，调用具体任务（目前仅ceval），保存对比报告。

 

**6）、src/merger/ -** **合并与量化**

model_merger.py：合并LoRA到基础模型，处理分片索引。

export.py：导出HF模型到GGUF并量化，调用llama.cpp工具。

convert_quant.py：使用llm-compressor进行GPTQ/AWQ量化，针对T4优化。

 

**7）、src/server/ -** **推理服务**

config.py：服务器配置类，从环境变量或默认值加载，针对T4优化。

model_manager.py：模型管理器（单例），负责模型加载/卸载、显存清理、提供生成上下文。

chat_system.py：核心推理引擎，加载模型，构建提示，生成响应（流式/非流式），解析思维链，管理会话历史。

inference_server.py：FastAPI应用，定义API端点，处理请求，调用model_manager。

schemas.py：Pydantic请求/响应模型。

streaming.py：流式响应生成器，处理SSE格式。

 

**8）、src/chat/ -** **客户端**

chat.py：CLI客户端，通过API调用服务，支持流式和颜色区分。

web_ui.py：Web UI服务，代理请求到推理服务，提供HTML界面。

 

**9）、src/config/ -** **配置**

settings.py：全局静态配置（模型路径、API地址等）。

 

**10）、src/utils/ -** **工具函数**

helpers.py：各类辅助函数：资源监控、内存使用、日志设置、数据集统计、响应清理、领域验证、复制配置文件等。

formatter.py：响应格式化器，去除think标签，应用markdown和领域高亮（当前未使用，可能用于客户端展示）。



## 5 步走：从数据到部署



### ① SFT 微调

```
CUDA_VISIBLE_DEVICES=1 python cli.py sft \
 --model /home/yaoxp/models/Qwen3-14B/ \
 --domain medical \
 --dataset "datasets/男科2w-fix.json,datasets/儿科3w-fix.json,datasets/妇产科2w-fix.json,datasets/medical_cn2w-fix.json" \
 --dataset_format alpaca \
 --mixing_strategy weighted \
 --epochs 3 \
 --max_seq_length 4096 \
 --batch_size 1 \
 --accumulation_steps 2 \
 --dataloader_workers 4 \
 --learning_rate 3e-5 \
 --no_packing \
 --lr_scheduler_type cosine \
 --save_steps 100 \
 --logging_steps 10 \
 --output_dir output/sft-qwen3-14b \
 --experiments_root output/experiments 
 
该指令提供了多个微调文件，并划分了权重，并且支持实验室隔离以及继续执行模式
如果想简单拼接所有数据（不按权重），可以改为：
--mixing_strategy concat \ # 去掉 --dataset_weights 参数

控制权重和微调数量
--dataset_weights "0.2,0.2,0.3,0.3" \
--data_limit 2000 \

中途微调断开，需要在最后save处继续执行
--resume auto
```



### ② 领域验证

```
使用多维度的评估方案（医疗微调方案采用）
python cli.py validate \
 --model /home/yaoxp/models/Qwen3-14B/ \
 --adapter output/experiments/sft_medical_004_0213_171642/final_adapter/ \
 --dataset datasets/男科2w-fix.json \
 --dataset_format alpaca \
 --max_samples 5 \
 --domain medical \
 --advanced
```



### ③ 评估（C-Eval）

\

```
# 如果未指定 --save_dir，程序会自动将结果保存到实验目录下的 evaluation/ 子目录中。
CUDA_VISIBLE_DEVICES=1 python cli.py evaluate \
 --task ceval \
 --model /home/yaoxp/models/Qwen3-14B/ \
 --adapter output/experiments/sft_medical_004_0213_171642/final_adapter/ \
 --task_dir datasets/ceval-exam \
 --n_shot 10 \
 --lang zh \
 --max_seq_length 4096 \
 --temperature 0.7 \
 --top_p 0.9 \
 --top_k 50 \
 --max_new_tokens 10 \
 --save_dir output/sft-qwen3-14b/evaluation_results
```

 

### ④ 权重合并 & 量化

```
合并 LoRA → 完整模型
python cli.py merge \
 --model /home/yaoxp/models/Qwen3-14B \
 --adapter output/experiments/sft_medical_002_0305_194002/final_adapter/ \
 --output output/sft-qwen3-14b/merged_model \
 --max_shard_size 2GB \
 --dtype float16

转换为 GGUF
cd /home/yaoxp/work/sft/uf8/output/sft-qwen3-14b/merged_model/
/home/yaoxp/work/llama.cpp/convert_hf_to_gguf.py --outfile qwen3-14b.gguf ./

格式转换-gptq
python src/merger/convert_quant.py \
 --model_path output/sft-qwen3-14b/merged_model \
 --output_dir output/sft-qwen3-14b/gptq4_model \
 --quant_scheme GPTQ \
 --bits 4 \
 --calib_samples 32 \
 --max_seq_len 128
```



### ⑤ 对话

启动推理服务和对话

```
# 方式1：启动LLM推理服务（端口12001），启动服务后，可以使用控制台对话
cd /home/yaoxp/work/sft/unsloth-factory08/scripts
./start_server.sh
终端访问：python -m src.chat.chat


\# 方式2：启动Web UI（端口8080），启动后，可以使用web浏览器对话
cd /home/yaoxp/work/sft/unsloth-factory08/scripts
./start_web_ui.sh
浏览器访问 ：http://172.16.0.93:8080（改成自己的ip）
```



启动服务

![](.\img\server.png)

console对话

![](.\img\chat.png)

web对话

![](.\img\web.png)

## 🔍 最后

自此从准备->微调->验证->评估->对话->格式转换，这一系列工程处理完毕，验证在极小尺寸的显卡中也能微调中等尺寸的大模型。并且对比验证了axolotl也能完成相同的事情，并且还有更多高级功能等待解锁；

与此同时，基于2次微调数据的校验对比，发现提前准备的微调数据质量不高，反而引起大模型推理能力下降，这里再次确认小批量高质量微调数据的效果要远大于大批量低质量的微调数据的效果。这是因为这批数据之前是没有cot的，是通过LLM反推cot数据，但推理的信息和质量没有达到医学问询标准，导致推理效果下降的原因。

 

另外

关于不同尺寸大模型在微调后效果对比，在之前的博客可以查询到

《Unsloth-Factory : 微调qwen3-14b大模型》 https://zhuanlan.zhihu.com/p/1938708116402341251

 

关于如何把gguf转为ollama部署，可以参考

《微调框架:Unsloth-factory》 https://zhuanlan.zhihu.com/p/1931426201798414500







 