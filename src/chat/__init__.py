#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聊天客户端模块
提供CLI和Web两种界面，均通过REST API调用推理服务
"""

# 注意：不再直接导出ChatSystem，它现在是server模块的依赖
# 如需使用核心推理，应从src.server.model_manager导入

__all__ = []  # 简化，不自动导出