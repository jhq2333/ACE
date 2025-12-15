#!/usr/bin/env python3
"""
检查webhook服务器的端点配置
"""
import requests
import json

def check_endpoints():
    """检查webhook服务器的端点"""
    base_url = "http://localhost:8000"
    
    print("检查webhook服务器端点...")
    
    # 检查根路径
    try:
        response = requests.get(base_url, timeout=5)
        print(f"根路径 (/): {response.status_code}")
    except Exception as e:
        print(f"无法访问根路径: {str(e)}")
        return
    
    # 检查OpenAPI文档
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"API文档 (/docs): {response.status_code}")
    except Exception as e:
        print(f"无法访问API文档: {str(e)}")
    
    # 检查OpenAPI JSON
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi_spec = response.json()
            print(f"OpenAPI规范 (/openapi.json): {response.status_code}")
            
            # 检查webhook端点
            if "/webhook" in openapi_spec.get("paths", {}):
                print("✓ /webhook 端点已定义")
                webhook_spec = openapi_spec["paths"]["/webhook"]
                if "post" in webhook_spec:
                    print("✓ /webhook POST 方法已定义")
                else:
                    print("✗ /webhook POST 方法未定义")
            else:
                print("✗ /webhook 端点未定义")
            
            # 检查stream端点
            if "/stream" in openapi_spec.get("paths", {}):
                print("✓ /stream 端点已定义")
            else:
                print("✗ /stream 端点未定义")
        else:
            print(f"无法获取OpenAPI规范: {response.status_code}")
    except Exception as e:
        print(f"获取OpenAPI规范时出错: {str(e)}")
    
    # 检查webhook端点（不带认证）
    try:
        response = requests.post(f"{base_url}/webhook", json={}, timeout=5)
        print(f"webhook端点 (/webhook): {response.status_code}")
        if response.status_code == 403:
            print("✓ webhook端点存在但需要认证")
        elif response.status_code == 404:
            print("✗ webhook端点不存在")
        else:
            print(f"webhook端点响应: {response.text}")
    except Exception as e:
        print(f"无法访问webhook端点: {str(e)}")

if __name__ == "__main__":
    check_endpoints()