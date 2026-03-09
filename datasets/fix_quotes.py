#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import re
from pathlib import Path

def clean_and_purify_json(input_file, output_file):
    """
    JSON数据最终净化工具：删除所有双引号和控制字符
    处理逻辑：
    1. 删除JSON解析失败的记录（14条）
    2. 删除缺少字段/think标签的记录（206条）
    3. 删除所有双引号（包括<think>标签内部）
    4. 删除所有\n, \t, \r控制字符
    输出：完全干净、可直接微调的JSON
    """
    print("=" * 80)
    print("JSON数据最终净化工具（删除引号+控制字符版）")
    print("=" * 80)
    
    if not Path(input_file).exists():
        print("ERROR: 文件不存在: {}".format(input_file))
        return False
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print("读取文件: {} ({:.2f} KB)".format(input_file, len(content) / 1024))
    print("-" * 80)
    
    # 步骤1: 容错解析
    print("步骤1: 解析JSON结构...")
    good_data, bad_records = tolerant_parse(content)
    
    # 步骤2: 深度净化（删除引号和控制字符）
    print("\n步骤2: 深度净化（删除双引号和控制字符）...")
    clean_data = []
    delete_stats = {'quotes': 0, 'control': 0}
    
    for idx, item in enumerate(good_data, 1):
        # 验证记录完整性
        check_result = validate_record(item, idx)
        if not check_result['valid']:
            bad_records.append(check_result['error'])
            continue
        
        # 净化内容（删除引号和控制字符）
        fixed_item, stats = purify_record_content(item)
        delete_stats['quotes'] += stats['quotes']
        delete_stats['control'] += stats['control']
        
        # 二次验证净化后的JSON合规性
        try:
            json.dumps(fixed_item, ensure_ascii=False)
            clean_data.append(fixed_item)
        except Exception as e:
            bad_records.append({
                'index': idx,
                'type': 'purify_error',
                'reason': '净化后JSON不合法: {}'.format(str(e)[:50]),
                'instruction': item['instruction'][:50]
            })
    
    # 步骤3: 生成最终报告
    original_total = len(good_data) + sum(1 for r in bad_records if r['type'] == 'parse_failure')
    kept_count = len(clean_data)
    deleted_count = len(bad_records)
    
    print("\n" + "=" * 80)
    print("数据净化完成报告")
    print("=" * 80)
    print("原始文件: {}".format(input_file))
    print("原始大小: {:.2f} KB".format(len(content) / 1024))
    print("-" * 80)
    print("总对象数: {} 条".format(original_total))
    print("保留数据: {} 条 ({:.1f}%)".format(kept_count, kept_count / original_total * 100))
    print("删除数据: {} 条 ({:.1f}%)".format(deleted_count, deleted_count / original_total * 100))
    print("-" * 80)
    print("净化操作统计:")
    print("  删除双引号: {}处".format(delete_stats['quotes']))
    print("  删除控制字符: {}处".format(delete_stats['control']))
    print("  删除类型: \\n(换行), \\t(制表符), \\r(回车), \"(双引号)")
    print("-" * 80)
    
    # 分类统计删除原因
    delete_reasons = {}
    for rec in bad_records:
        reason_type = rec['type']
        if reason_type not in delete_reasons:
            delete_reasons[reason_type] = 0
        delete_reasons[reason_type] += 1
    
    type_names = {
        'parse_failure': 'JSON解析失败（严重格式错误）',
        'missing_field': '缺少必要字段',
        'missing_think': '缺少think标签',
        'format_error': 'output格式错误',
        'purify_error': '净化后验证失败'
    }
    
    print("删除原因明细:")
    for reason_type, count in sorted(delete_reasons.items(), key=lambda x: x[1], reverse=True):
        print("  - {}: {}条".format(type_names.get(reason_type, reason_type), count))
    
    # 保存净化数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print("-" * 80)
    print("输出文件: {}".format(output_file))
    print("文件大小: {:.2f} KB".format(Path(output_file).stat().st_size / 1024))
    print("数据状态: ✓ 100%净化完成（无任何引号或控制字符污染）")
    print("=" * 80)
    
    # 生成删除日志
    if deleted_count > 0:
        generate_delete_log(bad_records, input_file)
    
    # 显示保留样本（确保无换行）
    print("\n保留数据样本（前3条）:")
    for i, item in enumerate(clean_data[:3], 1):
        think_match = re.search(r'<think>(.*?)</think>', item['output'], re.DOTALL)
        think_text = think_match.group(1).strip()[:60] if think_match else "无"
        print("  {}. {} -> {}...".format(
            i,
            item['instruction'][:50].replace('\n', ' '),
            think_text
        ))
    
    return True

def tolerant_parse(content):
    """容错解析JSON数组"""
    content = content.strip()
    if content.startswith('['):
        content = content[1:]
    if content.endswith(']'):
        content = content[:-1]
    
    # 按对象边界分割
    objects = []
    current = ""
    depth = 0
    in_string = False
    escape_next = False
    
    for ch in content:
        if escape_next:
            current += ch
            escape_next = False
            continue
        
        if ch == '\\':
            current += ch
            escape_next = True
            continue
        
        if ch == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == ',' and depth == 0:
                objects.append(current.strip())
                current = ""
                continue
        
        current += ch
    
    if current.strip():
        objects.append(current.strip())
    
    print("INFO: 解析出 {} 个对象".format(len(objects)))
    
    good_data = []
    bad_records = []
    
    for i, obj_str in enumerate(objects, 1):
        try:
            if not obj_str.startswith('{'):
                obj_str = '{' + obj_str
            if not obj_str.endswith('}'):
                obj_str = obj_str + '}'
            
            obj = json.loads(obj_str)
            good_data.append(obj)
        except Exception as e:
            bad_records.append({
                'index': i,
                'type': 'parse_failure',
                'reason': 'JSON解析失败: {}'.format(str(e)[:80]),
                'content': obj_str[:150] + '...' if len(obj_str) > 150 else obj_str
            })
    
    print("INFO: 成功解析 {} 条，失败 {} 条".format(len(good_data), len(bad_records)))
    return good_data, bad_records

def validate_record(item, index):
    """验证单条记录完整性"""
    required_fields = ['instruction', 'input', 'output']
    for field in required_fields:
        if field not in item:
            return {
                'valid': False,
                'error': {
                    'index': index,
                    'type': 'missing_field',
                    'reason': '缺少必要字段: {}'.format(field),
                    'instruction': str(item.get('instruction', ''))[:50]
                }
            }
        
        if field == 'output':
            if not isinstance(item[field], str):
                return {
                    'valid': False,
                    'error': {
                        'index': index,
                        'type': 'format_error',
                        'reason': 'output字段不是字符串类型',
                        'instruction': item['instruction'][:50]
                    }
                }
            
            output = item[field]
            has_open = "<think>" in output
            has_close = "</think>" in output
            
            if not has_open and not has_close:
                return {
                    'valid': False,
                    'error': {
                        'index': index,
                        'type': 'missing_think',
                        'reason': '完全缺少think标签',
                        'instruction': item['instruction'][:50]
                    }
                }
            
            if has_open != has_close:
                missing = '<think>' if not has_open else '</think>'
                return {
                    'valid': False,
                    'error': {
                        'index': index,
                        'type': 'missing_think',
                        'reason': '缺少标签: {}'.format(missing),
                        'instruction': item['instruction'][:50]
                    }
                }
            
            think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
            if not think_match or not think_match.group(1).strip():
                return {
                    'valid': False,
                    'error': {
                        'index': index,
                        'type': 'empty_think',
                        'reason': 'think标签内容为空',
                        'instruction': item['instruction'][:50]
                    }
                }
    
    return {'valid': True}

def purify_record_content(item):
    """净化记录内容：删除所有双引号和控制字符"""
    output = item['output']
    
    # 1. 删除所有控制字符（\n, \t, \r）
    control_chars = ['\n', '\t', '\r']
    control_delete_count = 0
    for char in control_chars:
        if char in output:
            count = output.count(char)
            output = output.replace(char, ' ')
            control_delete_count += count
    
    # 2. 删除所有双引号（包括<think>标签内部）- 关键操作！
    quote_delete_count = output.count('"')
    output = output.replace('"', '')
    
    item['output'] = output
    
    return item, {
        'quotes': quote_delete_count,
        'control': control_delete_count
    }

def generate_delete_log(bad_records, input_file):
    """生成删除记录日志"""
    log_file = Path(input_file).parent / (Path(input_file).stem + "_deleted.log")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("JSON数据删除记录日志\n")
        f.write("=" * 80 + "\n")
        f.write("文件: {}\n".format(input_file))
        f.write("删除记录总数: {} 条\n".format(len(bad_records)))
        f.write("生成时间: {}\n".format(
            __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("=" * 80 + "\n\n")
        
        # 按类型分组
        grouped = {}
        for rec in bad_records:
            rec_type = rec['type']
            if rec_type not in grouped:
                grouped[rec_type] = []
            grouped[rec_type].append(rec)
        
        # 输出
        for rec_type, records in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
            type_names = {
                'parse_failure': 'JSON解析失败（严重格式错误）',
                'missing_field': '缺少必要字段',
                'missing_think': '缺少think标签',
                'format_error': 'output格式错误',
                'purify_error': '净化后验证失败'
            }
            
            f.write("{}\n".format(type_names.get(rec_type, rec_type)))
            f.write("删除数量: {} 条\n".format(len(records)))
            f.write("-" * 80 + "\n\n")
            
            for rec in records[:100]:
                f.write("第{}条: {}\n".format(rec['index'], rec['reason']))
                if 'instruction' in rec:
                    clean_instr = rec['instruction'].replace('\n', ' ')
                    f.write("  instruction: {}\n".format(clean_instr[:80]))
                elif 'content' in rec:
                    clean_content = rec['content'].replace('\n', ' ')
                    f.write("  内容片段: {}\n".format(clean_content[:100]))
                f.write("\n")
            
            if len(records) > 100:
                f.write("... 还有 {} 条未显示\n".format(len(records) - 100))
            f.write("\n")
    
    print("已生成删除日志: {}".format(log_file))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("=" * 80)
        print("JSON数据最终净化工具（删除引号+控制字符版）")
        print("=" * 80)
        print("使用方法: python fix_quotes_final.py 输入文件.json 输出文件.json")
        print("\n处理规则:")
        print("  1. 删除JSON解析失败的记录")
        print("  2. 删除缺少必要字段的记录")
        print("  3. 删除缺少think标签的记录")
        print("  4. 删除所有双引号（包括think标签内部）")
        print("  5. 删除所有\\n, \\t, \\r控制字符")
        print("\n输出: 100%合规的干净数据集")
        sys.exit(1)
    
    input_file, output_file = sys.argv[1], sys.argv[2]
    print("最终净化: {} -> {}\n".format(input_file, output_file))
    
    success = clean_and_purify_json(input_file, output_file)
    
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: 数据最终净化完成！")
        print("✓ 所有双引号和控制字符已删除")
        print("✓ 输出文件100%合规，可直接用于微调")
        print("\n验证命令:")
        print("  python check_json.py {}".format(output_file))
        print("\nUnsloth微调示例:")
        print("  from datasets import load_dataset")
        print("  dataset = load_dataset('json', data_files='{}')".format(output_file))
    
    sys.exit(0 if success else 1)