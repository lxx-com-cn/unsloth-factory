import json
import sys

def convert_json_format(source_file, target_file):
    """
    将原始JSON格式转换为新的Alpaca格式
    
    转换规则：
    - instruction: 原始JSON中的system字段
    - input: 原始JSON中的instruction字段
    - output: 原始JSON中的output字段
    """
    
    try:
        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        # 初始化目标数据列表
        target_data = []
        
        # 遍历源数据中的每个对象
        for item in source_data:
            # 创建新的对象
            new_item = {
                "instruction": item.get("system", ""),
                "input": item.get("instruction", ""),
                "output": item.get("output", "")
            }
            target_data.append(new_item)
        
        # 写入目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！共处理 {len(target_data)} 条数据")
        print(f"源文件: {source_file}")
        print(f"目标文件: {target_file}")
        
        # 显示第一条转换后的数据作为示例
        if target_data:
            print("\n转换示例（第一条数据）:")
            print(json.dumps(target_data[0], ensure_ascii=False, indent=2))
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {source_file}")
        return False
    except json.JSONDecodeError:
        print(f"错误：{source_file} 不是有效的JSON格式")
        return False
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("使用方法: python alpaca_trans.py 源文件.json 目标文件.json")
        print("示例: python alpaca_trans.py source.json target.json")
        sys.exit(1)
    
    # 获取命令行参数
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    
    # 执行转换
    convert_json_format(source_file, target_file)