# src/merger/export.py
import os
import sys
import logging
import subprocess

logger = logging.getLogger(__name__)

def export_model(args):
    """
    Handles the export and quantization of models using llama.cpp tools.
    """
    # --- Validate Inputs ---
    if not args.llama_cpp:
        logger.error("必须提供 --llama_cpp 参数指定 llama.cpp 根目录路径。")
        return False
    if not os.path.isdir(args.llama_cpp):
        logger.error(f"指定的 llama.cpp 路径不存在或不是目录: {args.llama_cpp}")
        return False

    if not args.model:
        logger.error("必须提供 --model 参数指定 HF 合并模型的路径。")
        return False
    if not os.path.isdir(args.model):
        logger.error(f"指定的 HF 模型路径不存在或不是目录: {args.model}")
        return False

    llama_cpp_root = os.path.abspath(args.llama_cpp)
    hf_model_path = os.path.abspath(args.model)

    # --- Conversion Step ---
    conversion_success = True
    if args.gguf:
        # Determine paths for conversion script and output
        convert_script_path = os.path.join(llama_cpp_root, "convert_hf_to_gguf.py")
        if not os.path.isfile(convert_script_path):
            logger.error(f"未找到转换脚本: {convert_script_path}")
            return False

        gguf_output_path = os.path.abspath(args.gguf)
        # Ensure the output directory for the GGUF file exists
        gguf_output_dir = os.path.dirname(gguf_output_path)
        os.makedirs(gguf_output_dir, exist_ok=True)

        logger.info("开始执行 HF 到 GGUF 转换...")
        logger.info(f"  脚本路径: {convert_script_path}")
        logger.info(f"  HF 模型路径: {hf_model_path}")
        logger.info(f"  GGUF 输出路径: {gguf_output_path}")

        # Prepare command - assuming convert script takes input dir and --outfile
        cmd_convert = [
            sys.executable, # Use the same Python interpreter
            convert_script_path,
            hf_model_path, # Input HF model directory
            "--outfile", gguf_output_path # Specify output GGUF file path
        ]

        logger.info(f"执行命令: {' '.join(cmd_convert)}")

        try:
            # Use Popen for potentially long-running process and better control
            process = subprocess.Popen(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Log output in real-time
            for line in process.stdout:
                logger.info(f"[convert_hf_to_gguf] {line.strip()}")
            process.wait() # Wait for completion

            if process.returncode == 0:
                logger.info("HF 到 GGUF 转换成功完成。")
            else:
                logger.error(f"HF 到 GGUF 转换失败，退出码: {process.returncode}")
                return False
        except Exception as e:
            logger.error(f"执行转换命令时发生未知错误: {e}")
            return False
    else:
        logger.info("未提供 --gguf 参数，跳过 HF 到 GGUF 转换步骤。")
        conversion_success = False
        # If quantization is requested, gguf path is needed.
        if args.quant_method or args.quant_gguf:
             logger.warning("提供了量化参数但缺少 --gguf，将跳过量化步骤。")


    # --- Quantization Step ---
    if args.quant_method and args.gguf and args.quant_gguf and conversion_success:
        # Determine paths for quantize tool and files
        quantize_tool_path = os.path.join(llama_cpp_root, "build", "bin", "llama-quantize")
        # Check alternative common path if build/bin doesn't exist
        if not os.path.isfile(quantize_tool_path):
             quantize_tool_path = os.path.join(llama_cpp_root, "llama-quantize") # Sometimes built directly in root
        if not os.path.isfile(quantize_tool_path):
            logger.error(f"未找到量化工具: {quantize_tool_path} (尝试了 build/bin 和根目录)")
            return False

        input_gguf_path = os.path.abspath(args.gguf) # Use the GGUF file from conversion or user input
        if not os.path.isfile(input_gguf_path):
            logger.error(f"要量化的 GGUF 文件不存在: {input_gguf_path}")
            return False

        quant_gguf_output_path = os.path.abspath(args.quant_gguf)
        quant_output_dir = os.path.dirname(quant_gguf_output_path)
        os.makedirs(quant_output_dir, exist_ok=True)

        quant_method = args.quant_method

        logger.info("开始执行 GGUF 量化...")
        logger.info(f"  量化工具路径: {quantize_tool_path}")
        logger.info(f"  输入 GGUF 路径: {input_gguf_path}")
        logger.info(f"  量化方法: {quant_method}")
        logger.info(f"  量化后 GGUF 输出路径: {quant_gguf_output_path}")

        # Prepare command
        cmd_quantize = [
            quantize_tool_path,
            input_gguf_path,
            quant_gguf_output_path,
            quant_method
        ]

        logger.info(f"执行命令: {' '.join(cmd_quantize)}")

        try:
            # Use Popen for potentially long-running process and better control
            process = subprocess.Popen(cmd_quantize, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Log output in real-time
            for line in process.stdout:
                logger.info(f"[llama-quantize] {line.strip()}")
            process.wait() # Wait for completion

            if process.returncode == 0:
                logger.info(f"GGUF 量化 ({quant_method}) 成功完成。")
            else:
                logger.error(f"GGUF 量化失败，退出码: {process.returncode}")
                return False
        except Exception as e:
            logger.error(f"执行量化命令时发生未知错误: {e}")
            return False
    elif args.quant_method or args.quant_gguf:
        if conversion_success: # Only warn about quantization if conversion was attempted/succeeded
            logger.info("未提供完整的量化参数 (--quant_method 和 --quant_gguf)，跳过量化步骤。")
    else:
         logger.info("未请求量化步骤。")


    logger.info("导出流程完成。")
    return True
