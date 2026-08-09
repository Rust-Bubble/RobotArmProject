#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import fnmatch

# ================== 配置项 ==================
OUTPUT_FILE = "all_files_summary.txt"      # 汇总输出文件名

# 需要忽略的目录名（无论出现在哪个层级，都会整个跳过）
IGNORE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "robot-arm-env", "env",
    ".vscode", ".idea", "data"
}

# 需要忽略的文件名模式（支持通配符 *）
IGNORE_FILE_PATTERNS = {
    "*.pyc", "*.pyo", "*.log", ".DS_Store", "Thumbs.db"
}

# 需要忽略的精确相对路径（相对于项目根目录）
IGNORE_SPECIFIC_PATHS = {
    "config/database.yaml", "config/database.local.yaml"
}

# 自动忽略本脚本自身和输出文件（无需手动添加）
IGNORE_FILES_AUTO = {OUTPUT_FILE, os.path.basename(__file__)}
# ===========================================

def is_binary_file(filepath):
    """简单判断是否为二进制文件（尝试读取少量字节，若包含空字节则视为二进制）"""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
        return False
    except:
        return True

def should_ignore_file(relpath, filename):
    """根据文件名、模式、精确路径判断是否应忽略该文件"""
    # 自动忽略脚本自身和输出文件
    if filename in IGNORE_FILES_AUTO or relpath in IGNORE_FILES_AUTO:
        return True

    # 检查文件名模式
    for pattern in IGNORE_FILE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True

    # 检查精确相对路径
    if relpath in IGNORE_SPECIFIC_PATHS:
        return True

    return False

def export_files(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("文件内容汇总导出\n")
        out.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("=" * 80 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 阻止递归进入被忽略的目录（原地修改 dirnames）
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                relpath = os.path.relpath(filepath, root_dir)

                # 检查是否应该忽略该文件
                if should_ignore_file(relpath, filename):
                    continue

                out.write(f"\n{'=' * 80}\n")
                out.write(f"文件: {relpath}\n")
                out.write(f"路径: {filepath}\n")
                out.write('-' * 80 + "\n")

                # 尝试读取文件内容
                try:
                    if is_binary_file(filepath):
                        out.write("[注意] 该文件可能是二进制文件，跳过内容显示。\n")
                    else:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            out.write(content)
                            if not content.endswith('\n'):
                                out.write('\n')
                except UnicodeDecodeError:
                    # 尝试用其他编码（如 GBK）再试一次
                    try:
                        with open(filepath, 'r', encoding='gbk') as f:
                            content = f.read()
                            out.write(content)
                            if not content.endswith('\n'):
                                out.write('\n')
                    except:
                        out.write("[错误] 无法解码该文件内容，跳过。\n")
                except Exception as e:
                    out.write(f"[错误] 读取文件失败: {e}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("导出结束\n")
        out.write("=" * 80 + "\n")

if __name__ == "__main__":
    root = "."
    if not os.path.exists(OUTPUT_FILE):
        export_files(root, OUTPUT_FILE)
        print(f"导出完成！汇总文件已保存为: {OUTPUT_FILE}")
    else:
        print(f"警告: {OUTPUT_FILE} 已存在，为防止覆盖，请先删除或重命名。")