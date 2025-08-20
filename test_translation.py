#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试有道翻译API连接
"""

import sys
import time

# 添加apidemo_2路径
sys.path.append('apidemo_2')
from utils.AuthV4Util import addAuthParams
from utils.WebSocketUtil import init_connection_with_params, send_binary_message

def test_connection():
    """测试API连接"""
    print("正在测试有道翻译API连接...")
    
    # 测试配置
    APP_KEY = '5a8045edc022aece'  # 请替换为您的应用ID
    APP_SECRET = '3KjduBFD5Onzgfmg6ryHZlgE7C05yFky'  # 请替换为您的应用密钥
    
    # 连接参数
    data = {
        'from': 'zh-CHS',
        'to': 'en',
        'format': 'wav',
        'channel': '1',
        'version': 'v1',
        'rate': '16000'
    }
    
    try:
        # 添加鉴权参数
        addAuthParams(APP_KEY, APP_SECRET, data)
        print("✓ 鉴权参数生成成功")
        
        # 创建连接
        print("正在建立WebSocket连接...")
        ws_client = init_connection_with_params(
            "wss://openapi.youdao.com/stream_speech_trans", 
            data
        )
        
        # 等待连接
        timeout = 10
        start_time = time.time()
        while not ws_client.return_is_connect():
            if time.time() - start_time > timeout:
                print("✗ 连接超时")
                return False
            time.sleep(0.1)
        
        print("✓ WebSocket连接建立成功")
        
        # 发送测试数据
        print("发送测试音频数据...")
        test_audio = b'\x00\x00' * 1000  # 1KB的静音数据
        send_binary_message(ws_client.ws, test_audio)
        
        # 发送结束信号
        print("发送结束信号...")
        end_message = "{\"end\": \"true\"}"
        ws_client.ws.send(end_message)
        
        # 等待处理
        time.sleep(2)
        
        print("✓ API连接测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 连接测试失败: {e}")
        return False

def test_auth():
    """测试鉴权参数生成"""
    print("正在测试鉴权参数生成...")
    
    try:
        APP_KEY = 'test_key'
        APP_SECRET = 'test_secret'
        
        params = {'test': 'value'}
        addAuthParams(APP_KEY, APP_SECRET, params)
        
        required_keys = ['appKey', 'salt', 'curtime', 'signType', 'sign']
        for key in required_keys:
            if key not in params:
                print(f"✗ 缺少鉴权参数: {key}")
                return False
        
        print("✓ 鉴权参数生成成功")
        return True
        
    except Exception as e:
        print(f"✗ 鉴权参数生成失败: {e}")
        return False

def main():
    print("=== 有道翻译API连接测试 ===\n")
    
    # 测试鉴权
    auth_ok = test_auth()
    print()
    
    # 测试连接
    conn_ok = test_connection()
    print()
    
    # 总结
    if auth_ok and conn_ok:
        print("🎉 所有测试通过！API配置正确。")
        print("您现在可以运行 realtime_translation_demo.py 进行实时翻译了。")
    else:
        print("❌ 部分测试失败，请检查配置。")
        if not auth_ok:
            print("- 检查鉴权参数生成")
        if not conn_ok:
            print("- 检查网络连接")
            print("- 验证API密钥")
            print("- 确认API服务状态")

if __name__ == "__main__":
    main()
