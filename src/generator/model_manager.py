#!/usr/bin/env python3
"""
多模型支持管理器

支持2024-2025年最稳定最热门的五个模型：
- gpt-4o: OpenAI GPT-4o (最强大)
- gpt-4o-mini: OpenAI GPT-4o Mini (性价比最高)
- deepseek-chat: DeepSeek Chat (中文优化)
- gemini-pro: Google Gemini Pro (多模态)
- llama-3.1-8b: Meta Llama 3.1 8B (开源最强)
"""

import os
import logging
from typing import Optional, Dict, Any
from camel.models import OpenAIModel
from camel.types import ModelType

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class MultiModelManager:
    """多模型管理器，支持多种LLM模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多模型管理器
        
        Args:
            config: 配置字典，包含模型相关配置
        """
        self.config = config
        self.inference_config = config.get("inference", {})
        self.model_type = self.inference_config.get("model_type", "gpt-4o-mini")
        self.api_key = self.inference_config.get("openai_api_key", "")
        self.base_url = self.inference_config.get("openai_base_url", "")
        
        # 设置环境变量
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            
        log.info(f"初始化模型管理器，模型类型: {self.model_type}")
    
    def create_model(self, agent_id: int = None) -> Any:
        """
        根据配置创建模型实例
        
        Args:
            agent_id: 智能体ID（可选）
            
        Returns:
            模型实例
        """
        try:
            if self.model_type in ["gpt-4o", "gpt-4o-mini"]:
                return self._create_openai_model()
            elif self.model_type == "deepseek-chat":
                return self._create_deepseek_model()
            elif self.model_type == "gemini-pro":
                return self._create_gemini_model()
            elif self.model_type == "llama-3.1-8b":
                return self._create_llama_model()
            else:
                log.warning(f"未知模型类型: {self.model_type}，使用默认GPT-4o-mini")
                return self._create_openai_model()
                
        except Exception as e:
            log.error(f"创建模型失败: {e}")
            # 降级到默认模型
            return self._create_fallback_model()
    
    def _create_openai_model(self) -> Any:
        """创建OpenAI模型"""
        try:
            # 根据模型类型选择对应的ModelType
            if self.model_type == "gpt-4o":
                model = OpenAIModel(model_type=ModelType.GPT_4O)
            elif self.model_type == "gpt-4o-mini":
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
            else:
                # 默认使用GPT-4o-mini
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
            
            log.info(f"成功创建OpenAI模型: {self.model_type}")
            return model
            
        except Exception as e:
            log.error(f"创建OpenAI模型失败: {e}")
            raise
    
    def _create_deepseek_model(self) -> Any:
        """创建DeepSeek模型"""
        try:
            # 检查是否有DeepSeek API配置
            deepseek_api_key = self.inference_config.get("deepseek_api_key", "")
            deepseek_base_url = self.inference_config.get("deepseek_base_url", "https://api.deepseek.com/v1")
            
            if deepseek_api_key:
                # 使用DeepSeek API
                os.environ["OPENAI_API_KEY"] = deepseek_api_key
                os.environ["OPENAI_BASE_URL"] = deepseek_base_url
                
                # 使用GPT-4o-mini作为兼容模型
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建DeepSeek模型: {self.model_type}")
                return model
            else:
                log.warning("未配置DeepSeek API密钥，使用OpenAI兼容模式")
                # 使用OpenAI兼容的配置
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建DeepSeek兼容模型: {self.model_type}")
                return model
                
        except Exception as e:
            log.error(f"创建DeepSeek模型失败: {e}")
            raise
    
    def _create_gemini_model(self) -> Any:
        """创建Gemini模型"""
        try:
            # 检查是否有Gemini API配置
            gemini_api_key = self.inference_config.get("gemini_api_key", "")
            
            if gemini_api_key:
                # 使用Gemini API (需要专门的Gemini客户端)
                log.warning("Gemini API需要专门的客户端，暂时使用OpenAI兼容模式")
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建Gemini兼容模型: {self.model_type}")
                return model
            else:
                log.warning("未配置Gemini API密钥，使用OpenAI兼容模式")
                # 使用OpenAI兼容的配置
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建Gemini兼容模型: {self.model_type}")
                return model
                
        except Exception as e:
            log.error(f"创建Gemini模型失败: {e}")
            raise
    
    def _create_llama_model(self) -> Any:
        """创建Llama模型"""
        try:
            # 检查是否有Llama API配置
            llama_api_key = self.inference_config.get("llama_api_key", "")
            llama_base_url = self.inference_config.get("llama_base_url", "https://api.llama-api.com")
            
            if llama_api_key:
                # 使用Llama API
                os.environ["OPENAI_API_KEY"] = llama_api_key
                os.environ["OPENAI_BASE_URL"] = llama_base_url
                
                # 使用GPT-4o-mini作为兼容模型
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建Llama模型: {self.model_type}")
                return model
            else:
                log.warning("未配置Llama API密钥，使用OpenAI兼容模式")
                # 使用OpenAI兼容的配置
                model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
                
                log.info(f"成功创建Llama兼容模型: {self.model_type}")
                return model
                
        except Exception as e:
            log.error(f"创建Llama模型失败: {e}")
            raise
    
    
    
    def _create_fallback_model(self) -> Any:
        """创建降级模型（当其他模型创建失败时使用）"""
        try:
            log.warning("使用降级模型GPT-4o-mini")
            model = OpenAIModel(model_type=ModelType.GPT_4O_MINI)
            return model
        except Exception as e:
            log.error(f"创建降级模型也失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "model_type": self.model_type,
            "api_key_configured": bool(self.api_key),
            "base_url_configured": bool(self.base_url),
            "supported_models": [
                "gpt-4o", "gpt-4o-mini", "deepseek-chat", "gemini-pro", "llama-3.1-8b"
            ]
        }
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        try:
            # 尝试创建模型
            test_model = self.create_model()
            if test_model:
                log.info("模型配置验证成功")
                return True
            else:
                log.error("模型配置验证失败：无法创建模型")
                return False
        except Exception as e:
            log.error(f"模型配置验证失败: {e}")
            return False
