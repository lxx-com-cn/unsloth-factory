import json
import sys
import re
from pathlib import Path

def contains_chinese(text):
    """检查文本是否包含中文字符"""
    if not text or not isinstance(text, str):
        return False
    
    # 检查中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def remove_english_items(source_file, target_file):
    """
    删除JSON中input和output字段都是英文的数据项
    
    规则：
    1. 如果input和output字段都不包含中文，则删除整个数据项
    2. 如果至少一个字段包含中文，则保留整个数据项
    3. 保留instruction字段不做检查（即使它是中文，如果内容都是英文，也要删除）
    """
    
    try:
        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("错误：JSON文件应该是一个数组")
            return False
        
        filtered_data = []
        removed_count = 0
        total_items = len(data)
        
        print(f"开始处理文件: {source_file}")
        print(f"总条目数: {total_items}")
        print("-" * 50)
        
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                print(f"警告：第{i}项不是字典格式，已跳过")
                continue
            
            # 获取input和output字段
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # 检查是否包含中文
            input_has_chinese = contains_chinese(str(input_text))
            output_has_chinese = contains_chinese(str(output_text))
            
            # 只要有一个字段包含中文，就保留整个数据项
            if input_has_chinese or output_has_chinese:
                filtered_data.append(item)
                if i <= 5:  # 显示前5个保留的条目
                    print(f"✓ 保留第{i}条: {item.get('instruction', '无instruction')[:80]}")
            else:
                removed_count += 1
                if i <= 5 or removed_count <= 5:  # 显示前5个删除的条目
                    instruction_preview = item.get("instruction", "无instruction")[:60]
                    print(f"✗ 删除第{i}条: {instruction_preview}")
                    if input_text:
                        input_preview = str(input_text)[:60] + ("..." if len(str(input_text)) > 60 else "")
                        print(f"   input: {input_preview}")
                    if output_text:
                        output_preview = str(output_text)[:60] + ("..." if len(str(output_text)) > 60 else "")
                        print(f"  output: {output_preview}")
        
        # 写入目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 50)
        print(f"处理完成！")
        print(f"原始条目数: {total_items}")
        print(f"处理后条目数: {len(filtered_data)}")
        print(f"删除的英文条目数: {removed_count}")
        print(f"删除比例: {removed_count/total_items*100:.2f}%")
        print(f"目标文件: {target_file}")
        
        # 显示处理前后的对比示例
        if data and filtered_data:
            print("\n处理前后对比示例:")
            
            # 找一个被删除的示例
            deleted_examples = []
            for i, item in enumerate(data):
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                if not contains_chinese(str(input_text)) and not contains_chinese(str(output_text)):
                    deleted_examples.append((i+1, item))
                    if len(deleted_examples) >= 1:
                        break
            
            if deleted_examples:
                idx, deleted_item = deleted_examples[0]
                print(f"被删除的示例（第{idx}条）:")
                print(json.dumps(deleted_item, ensure_ascii=False, indent=2)[:500])
                print("...")
            
            # 显示保留的示例
            if filtered_data:
                print(f"\n保留的示例（第1条）:")
                print(json.dumps(filtered_data[0], ensure_ascii=False, indent=2))
        
        # 生成统计报告
        generate_statistics_report(data, filtered_data, removed_count, source_file, target_file)
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {source_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误：{source_file} 不是有效的JSON格式: {e}")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_statistics_report(original_data, filtered_data, removed_count, source_file, target_file):
    """生成详细的统计报告"""
    
    print("\n" + "="*60)
    print("详细统计报告")
    print("="*60)
    
    total = len(original_data)
    kept = len(filtered_data)
    removed = removed_count
    
    print(f"1. 总体统计:")
    print(f"   原始数据量: {total} 条")
    print(f"   保留数据量: {kept} 条")
    print(f"   删除数据量: {removed} 条")
    print(f"   保留比例: {kept/total*100:.2f}%")
    print(f"   删除比例: {removed/total*100:.2f}%")
    
    # 分析删除原因
    chinese_only_in_input = 0
    chinese_only_in_output = 0
    chinese_in_both = 0
    no_chinese = 0
    
    for item in original_data:
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        input_has_chinese = contains_chinese(str(input_text))
        output_has_chinese = contains_chinese(str(output_text))
        
        if input_has_chinese and output_has_chinese:
            chinese_in_both += 1
        elif input_has_chinese and not output_has_chinese:
            chinese_only_in_input += 1
        elif not input_has_chinese and output_has_chinese:
            chinese_only_in_output += 1
        else:
            no_chinese += 1
    
    print(f"\n2. 中文分布统计:")
    print(f"   中英文混合数据: {chinese_in_both} 条 ({chinese_in_both/total*100:.2f}%)")
    print(f"   仅input含中文: {chinese_only_in_input} 条 ({chinese_only_in_input/total*100:.2f}%)")
    print(f"   仅output含中文: {chinese_only_in_output} 条 ({chinese_only_in_output/total*100:.2f}%)")
    print(f"   纯英文数据: {no_chinese} 条 ({no_chinese/total*100:.2f}%)")
    
    # 文件大小对比
    source_path = Path(source_file)
    target_path = Path(target_file)
    
    if source_path.exists() and target_path.exists():
        source_size = source_path.stat().st_size / 1024  # KB
        target_size = target_path.stat().st_size / 1024  # KB
        
        print(f"\n3. 文件大小对比:")
        print(f"   源文件大小: {source_size:.2f} KB")
        print(f"   目标文件大小: {target_size:.2f} KB")
        print(f"   大小减少: {source_size - target_size:.2f} KB ({((source_size - target_size)/source_size*100):.2f}%)")
    
    print("\n4. 处理规则:")
    print("   - 如果input或output字段包含中文，保留整个数据项")
    print("   - 如果input和output字段都不包含中文，删除整个数据项")
    print("   - instruction字段不做检查，仅用于参考")
    print("="*60)

def main():
    """主函数，处理命令行参数"""
    if len(sys.argv) != 3:
        print("英文数据清理工具")
        print("=" * 50)
        print("使用方法: python english_remove.py 源文件.json 目标文件.json")
        print("示例: python english_remove.py medical_o1_alpaca.json medical_o1_alpaca_chinese.json")
        print("\n功能说明:")
        print("1. 删除input和output字段都不包含中文的数据项")
        print("2. 保留至少一个字段包含中文的数据项")
        print("3. 生成详细的统计报告")
        sys.exit(1)
    
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    
    # 检查源文件是否存在
    if not Path(source_file).exists():
        print(f"错误：源文件不存在 {source_file}")
        sys.exit(1)
    
    # 确认目标文件是否已存在
    if Path(target_file).exists():
        print(f"警告：目标文件 {target_file} 已存在")
        response = input("是否覆盖？(y/n): ").strip().lower()
        if response != 'y':
            print("操作已取消")
            sys.exit(0)
    
    print("英文数据清理工具")
    print("=" * 50)
    
    # 执行处理
    success = remove_english_items(source_file, target_file)
    
    if not success:
        sys.exit(1)
    
    print("\n清理完成！中文数据集已准备好用于微调。")

if __name__ == "__main__":
    main()