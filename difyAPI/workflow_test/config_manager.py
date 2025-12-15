# config_manager.py
import json
import os

class ConfigManager:
    def __init__(self, config_path="difyAPI\\workflow_test\\config.json"):
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
    
    @property
    def api_url(self):
        return self._config['dify']['api_url']
    
    @property
    def api_key(self):
        # 优先从环境变量获取，其次从配置文件获取
        return os.getenv('DIFY_API_KEY', self._config['dify']['api_key'])
    
    def get_workflow_id(self, workflow_name):
        return self._config['workflows'].get(workflow_name)

# 创建全局配置实例
config = ConfigManager()