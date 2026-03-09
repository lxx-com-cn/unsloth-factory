import json
import sys
import re
from pathlib import Path

def check_json_format(file_path):
    """
    检查JSON格式和output字段的完整性
    
    检查内容：
    1. 文件是否为有效的JSON格式
    2. 是否为数组格式
    3. 每个对象是否包含必要的字段（instruction, input, output）
    4. output字段是否包含<think>和</think>标签
    5. output字段中是否有嵌套双引号（需要手工处理的问题）
    """
    
    print("JSON格式检查工具")
    print("=" * 60)
    
    try:
        # 检查文件是否存在
        if not Path(file_path).exists():
            print(f"错误：文件不存在 {file_path}")
            return False
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"检查文件: {file_path}")
        print(f"文件大小: {len(content) / 1024:.2f} KB")
        print("-" * 60)
        
        # 尝试解析JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON格式错误: {e}")
            print(f"错误位置: 第{e.lineno}行, 第{e.colno}列, 字符位置{e.pos}")
            
            # 显示错误附近的上下文
            lines = content.split('\n')
            
            print("\n错误附近的上下文:")
            # 显示错误行前后5行
            start_line = max(0, e.lineno - 6)
            end_line = min(len(lines), e.lineno + 5)
            
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if line_num == e.lineno else "    "
                print(f"{prefix}第{line_num:5d}行: {lines[i]}")
                
                # 如果是错误行，标记出错误列位置
                if line_num == e.lineno and e.colno > 0:
                    print(f"      {' ' * (e.colno + 7)}^ 错误可能在此处附近")
            
            return False
        
        print("✓ JSON格式验证通过")
        
        # 检查是否为数组
        if not isinstance(data, list):
            print("❌ JSON不是数组格式，应为列表")
            return False
        
        print(f"✓ 数据格式为数组，包含 {len(data)} 条记录")
        print("-" * 60)
        
        # 初始化统计信息
        total_items = len(data)
        missing_fields = []
        output_issues = []
        quote_issues = []  # 记录嵌套双引号问题
        valid_items = 0
        
        # 检查每条记录
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                print(f"❌ 第{i}条: 不是字典格式")
                missing_fields.append(i)
                continue
            
            # 检查必要字段
            issues = []
            
            # 检查instruction字段
            if "instruction" not in item:
                issues.append("缺少instruction字段")
            elif not item["instruction"] or not isinstance(item["instruction"], str):
                issues.append("instruction字段为空或非字符串")
            
            # 检查input字段（可以为空字符串）
            if "input" not in item:
                issues.append("缺少input字段")
            elif not isinstance(item.get("input", ""), str):
                issues.append("input字段非字符串")
            
            # 检查output字段
            if "output" not in item:
                issues.append("缺少output字段")
            else:
                output_text = item["output"]
                if not isinstance(output_text, str):
                    issues.append("output字段非字符串")
                else:
                    # 检查<think>和</think>标签
                    if "<think>" not in output_text:
                        issues.append("output缺少<think>标签")
                    elif "</think>" not in output_text:
                        issues.append("output缺少</think>标签")
                    elif output_text.count("<think>") != output_text.count("</think>"):
                        issues.append("<think>和</think>标签不匹配")
                    else:
                        # 提取think内容
                        think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
                        if think_match:
                            think_content = think_match.group(1).strip()
                            if not think_content:
                                issues.append("<think>标签内容为空")
                            elif len(think_content) < 5:  # 降低最小长度检查
                                issues.append(f"<think>内容过短 ({len(think_content)}字符)")
                            
                            # 检查output中是否有嵌套双引号
                            # 在整个output中查找未转义的双引号（不在JSON字符串边界处）
                            # 简单检查：查找双引号前面不是冒号或逗号+空格的情况
                            # 更准确的方法：检查双引号是否在字符串内容中
                            unescaped_quotes = []
                            
                            # 查找output中所有的双引号位置
                            for match in re.finditer(r'(?<!\\)"', output_text):
                                pos = match.start()
                                # 检查这个双引号是否在标签外部（不是在<think>或</think>标签名中）
                                if not (output_text[pos-7:pos] == "<think" or 
                                        output_text[pos-8:pos] == "</think"):
                                    # 检查前后字符，判断是否是字符串内容中的双引号
                                    if pos > 0 and pos < len(output_text) - 1:
                                        prev_char = output_text[pos-1]
                                        next_char = output_text[pos+1]
                                        # 如果双引号前后都是中文字符或字母，可能是在内容中
                                        if ((is_chinese(prev_char) or prev_char.isalpha() or prev_char in " ,，.。") and 
                                            (is_chinese(next_char) or next_char.isalpha() or next_char in " ,，.。！？")):
                                            unescaped_quotes.append(pos)
                            
                            if unescaped_quotes:
                                # 记录嵌套双引号问题
                                quote_issues.append((i, output_text, unescaped_quotes))
                        else:
                            issues.append("无法提取<think>内容")
            
            # 如果有问题，记录
            if issues:
                missing_fields.append(i)
                output_issues.append((i, issues))
                print(f"❌ 第{i}条: {', '.join(issues)}")
            else:
                valid_items += 1
        
        # 输出统计报告
        print("-" * 60)
        print("检查完成！")
        print(f"总记录数: {total_items}")
        print(f"有效记录数: {valid_items} ({valid_items/total_items*100:.1f}%)")
        print(f"格式问题记录数: {len(missing_fields)} ({len(missing_fields)/total_items*100:.1f}%)")
        print(f"嵌套双引号问题记录数: {len(quote_issues)} ({len(quote_issues)/total_items*100:.1f}%)")
        
        # 输出嵌套双引号问题的详细信息
        if quote_issues:
            print(f"\n⚠️  发现 {len(quote_issues)} 条记录有嵌套双引号问题（需要手工处理）:")
            print("=" * 60)
            
            for idx, (item_num, output_text, quote_positions) in enumerate(quote_issues[:20]):  # 最多显示20条
                print(f"\n[{idx+1}] 第{item_num}条记录 - 发现 {len(quote_positions)} 处嵌套双引号:")
                
                # 获取对应的完整数据项
                if item_num <= len(data):
                    item = data[item_num-1]
                    print(f"   instruction: {item.get('instruction', '')}")
                    print(f"   input: {item.get('input', '')}")
                
                # 显示output中嵌套双引号的位置
                print(f"   output (显示前500字符):")
                
                # 高亮显示有问题的部分
                output_preview = output_text[:500]
                if len(output_text) > 500:
                    output_preview += "..."
                
                # 标记出双引号位置
                marked_output = output_preview
                # 从后往前插入标记，避免位置变化
                for pos in sorted(quote_positions, reverse=True):
                    if pos < 500:  # 只标记预览范围内的
                        marked_output = marked_output[:pos] + "【问题双引号】" + marked_output[pos:pos+1] + "【结束】" + marked_output[pos+1:]
                
                print(f"   {marked_output}")
                
                # 显示具体的双引号上下文
                print(f"\n   具体问题位置:")
                for pos in quote_positions[:5]:  # 每个记录最多显示5个具体位置
                    start = max(0, pos - 30)
                    end = min(len(output_text), pos + 30)
                    context = output_text[start:end]
                    print(f"     位置{pos}: ...{context}...")
                
                print("-" * 40)
            
            if len(quote_issues) > 20:
                print(f"\n... 还有 {len(quote_issues)-20} 条有嵌套双引号问题的记录未显示")
        
        if missing_fields:
            print(f"\n格式问题记录的行号: {', '.join(map(str, missing_fields[:50]))}")
            if len(missing_fields) > 50:
                print(f"  ... 共{len(missing_fields)}条记录有格式问题")
            
            print("\n详细格式问题列表（显示前10条）:")
            for i, issues in output_issues[:10]:
                print(f"\n第{i}条: {', '.join(issues)}")
            
            if len(output_issues) > 10:
                print(f"\n  ... 还有{len(output_issues)-10}条有格式问题的记录")
        
        # 显示数据样本
        print("\n" + "=" * 60)
        print("数据样本检查（前3条有效记录）:")
        
        sample_count = 0
        for i, item in enumerate(data, 1):
            if i-1 not in missing_fields:  # 是有效记录
                print(f"\n--- 第{i}条样本 ---")
                print(f"instruction: {item.get('instruction', '')[:80]}...")
                print(f"input: {item.get('input', '')[:80]}...")
                
                output_text = item.get('output', '')
                if "<think>" in output_text and "</think>" in output_text:
                    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        print(f"<think>内容: {think_content[:100]}...")
                
                sample_count += 1
                if sample_count >= 3:
                    break
        
        # 输出建议
        print("\n" + "=" * 60)
        print("建议:")
        
        if missing_fields:
            print("1. 请修复格式问题的记录")
        
        if quote_issues:
            print("2. 请手工处理嵌套双引号问题:")
            print("   - 删除output字段中内容部分的双引号")
            print("   - 例如：主诉\"睾丸炎\" 改为 主诉睾丸炎")
            print("   - 例如：症状描述\"睾丸肿大、疼痛、阴囊皮肤红肿\" 改为 症状描述睾丸肿大、疼痛、阴囊皮肤红肿")
        
        if not missing_fields and not quote_issues:
            print("✓ 所有记录格式正确，可以用于微调")
        
        # 生成摘要报告
        print("\n" + "=" * 60)
        print("摘要报告:")
        print(f"文件: {file_path}")
        print(f"总记录: {total_items}")
        print(f"有效记录: {valid_items}")
        print(f"格式问题记录: {len(missing_fields)}")
        print(f"嵌套双引号问题记录: {len(quote_issues)}")
        
        if len(missing_fields) > 0 or len(quote_issues) > 0:
            print("状态: ⚠️  需要处理")
            
            # 生成问题报告文件
            generate_issue_report(file_path, data, missing_fields, quote_issues)
        else:
            print("状态: ✓ 通过检查")
            return True
        
        return False
        
    except Exception as e:
        print(f"检查过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def is_chinese(char):
    """检查字符是否为中文字符"""
    return '\u4e00' <= char <= '\u9fff'

def generate_issue_report(file_path, data, missing_fields, quote_issues):
    """生成问题报告文件"""
    try:
        report_file = Path(file_path).stem + "_issues.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("JSON文件问题报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"文件: {file_path}\n")
            f.write(f"检查时间: {sys.argv[0]}\n")
            f.write("=" * 60 + "\n\n")
            
            if missing_fields:
                f.write("一、格式问题记录\n")
                f.write("-" * 40 + "\n")
                f.write(f"共 {len(missing_fields)} 条记录有格式问题\n\n")
                
                for i in missing_fields[:100]:  # 最多显示100条
                    if i <= len(data):
                        item = data[i-1]
                        f.write(f"第{i}条记录:\n")
                        f.write(f"  instruction: {item.get('instruction', '')}\n")
                        f.write(f"  input: {item.get('input', '')}\n")
                        f.write(f"  output: {item.get('output', '')[:200]}...\n")
                        f.write("-" * 40 + "\n")
                
                if len(missing_fields) > 100:
                    f.write(f"... 还有 {len(missing_fields)-100} 条记录未显示\n\n")
            
            if quote_issues:
                f.write("\n二、嵌套双引号问题记录（需要手工处理）\n")
                f.write("-" * 40 + "\n")
                f.write(f"共 {len(quote_issues)} 条记录有嵌套双引号问题\n\n")
                
                for idx, (item_num, output_text, quote_positions) in enumerate(quote_issues[:50]):  # 最多显示50条
                    f.write(f"[{idx+1}] 第{item_num}条记录:\n")
                    
                    if item_num <= len(data):
                        item = data[item_num-1]
                        f.write(f"  instruction: {item.get('instruction', '')}\n")
                        f.write(f"  input: {item.get('input', '')}\n")
                    
                    f.write(f"  output (发现 {len(quote_positions)} 处嵌套双引号):\n")
                    
                    # 显示output，标记双引号位置
                    output_display = output_text
                    # 简单标记双引号
                    for pos in sorted(quote_positions, reverse=True):
                        output_display = output_display[:pos] + "→" + output_display[pos] + "←" + output_display[pos+1:]
                    
                    f.write(f"  {output_display[:300]}")
                    if len(output_display) > 300:
                        f.write("...")
                    f.write("\n")
                    
                    f.write(f"  需要手工删除的双引号位置: {quote_positions[:10]}\n")
                    f.write("-" * 40 + "\n")
                
                if len(quote_issues) > 50:
                    f.write(f"... 还有 {len(quote_issues)-50} 条记录未显示\n")
            
            f.write("\n三、处理建议\n")
            f.write("-" * 40 + "\n")
            f.write("1. 格式问题：检查并修复JSON格式\n")
            f.write("2. 嵌套双引号问题：\n")
            f.write("   - 打开原始JSON文件\n")
            f.write("   - 根据报告中的行号找到对应记录\n")
            f.write("   - 在output字段中，删除内容部分的双引号\n")
            f.write("   - 例如：将 主诉\"睾丸炎\" 改为 主诉睾丸炎\n")
            f.write("   - 注意：不要删除<think>和</think>标签\n")
        
        print(f"✓ 已生成详细问题报告: {report_file}")
        
    except Exception as e:
        print(f"生成问题报告时出错: {e}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("JSON格式检查工具")
        print("=" * 50)
        print("使用方法: python check_json.py 源文件.json")
        print("示例: python check_json.py medical_o1_alpaca_chinese.json")
        print("\n检查内容:")
        print("1. JSON格式有效性")
        print("2. 数组结构")
        print("3. instruction, input, output字段完整性")
        print("4. output字段是否包含<think>和</think>标签")
        print("5. output字段中是否有嵌套双引号（需要手工处理）")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # 执行检查
    is_valid = check_json_format(file_path)
    
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()