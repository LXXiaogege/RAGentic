# -*- coding: utf-8 -*-
"""安全工具模块 - API Key 加密、输入验证"""

import base64
import hashlib
import os
import re
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityManager:
    """安全管理器 - 处理加密和验证"""

    def __init__(self, secret_key: Optional[str] = None):
        """
        初始化安全管理器

        Args:
            secret_key: 加密密钥，如果为 None 则生成新密钥
        """
        if secret_key:
            self.key = self._derive_key(secret_key)
        else:
            self.key = Fernet.generate_key()

        self.fernet = Fernet(self.key)

    def _derive_key(self, password: str) -> bytes:
        """从密码派生加密密钥"""
        salt_str = os.environ.get("ENCRYPTION_SALT", "ragentic_salt_v1")
        salt = salt_str.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, plaintext: str) -> str:
        """加密字符串"""
        encrypted = self.fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, ciphertext: str) -> str:
        """解密字符串"""
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.fernet.decrypt(encrypted)
        return decrypted.decode()

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """哈希 API Key（用于日志和显示）"""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        验证 API Key 格式

        Args:
            api_key: API Key 字符串

        Returns:
            bool: 是否有效
        """
        if not api_key or len(api_key) < 10:
            return False

        # 检查是否包含明显的无效字符
        if "<" in api_key or ">" in api_key or "your_" in api_key.lower():
            return False

        return True

    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """
        清理用户输入，防止 XSS 和注入攻击

        Args:
            text: 输入文本
            max_length: 最大长度

        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""

        # 限制长度
        text = text[:max_length]

        # 移除危险字符
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # Script 标签
            r"javascript:",  # JavaScript 协议
            r"on\w+\s*=",  # 事件处理器
            r"<iframe[^>]*>",  # iframe
        ]

        for pattern in dangerous_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """
        验证用户查询

        Args:
            query: 查询文本

        Returns:
            tuple[bool, str]: (是否有效，错误信息)
        """
        if not query or not query.strip():
            return False, "查询不能为空"

        if len(query) > 5000:
            return False, "查询过长，最大长度 5000 字符"

        # 检查 SQL 注入特征
        sql_patterns = [
            r"\bDROP\s+TABLE\b",
            r"\bDELETE\s+FROM\b",
            r"\bINSERT\s+INTO\b",
            r"\bUPDATE\s+.*\s+SET\b",
        ]

        for pattern in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "查询包含不被允许的内容"

        return True, ""


# 全局安全实例
_security_manager: Optional[SecurityManager] = None


def get_security_manager(secret_key: Optional[str] = None) -> SecurityManager:
    """获取全局安全管理器实例"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(secret_key)
    return _security_manager


def encrypt_sensitive_data(data: str, secret_key: Optional[str] = None) -> str:
    """加密敏感数据"""
    manager = get_security_manager(secret_key)
    return manager.encrypt(data)


def decrypt_sensitive_data(ciphertext: str, secret_key: Optional[str] = None) -> str:
    """解密敏感数据"""
    manager = get_security_manager(secret_key)
    return manager.decrypt(ciphertext)
