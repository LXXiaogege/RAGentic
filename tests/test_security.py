# -*- coding: utf-8 -*-
"""安全模块单元测试"""

from src.utils.security import SecurityManager


class TestSecurityManager:
    """安全管理器测试"""

    def test_encrypt_decrypt(self):
        """测试加密解密"""
        manager = SecurityManager()
        plaintext = "sk-1234567890abcdef"

        encrypted = manager.encrypt(plaintext)
        decrypted = manager.decrypt(encrypted)

        assert decrypted == plaintext
        assert encrypted != plaintext

    def test_encrypt_decrypt_with_custom_key(self):
        """测试自定义密钥加密解密"""
        manager = SecurityManager(secret_key="my_secret_key_123")
        plaintext = "sensitive_data"

        encrypted = manager.encrypt(plaintext)
        decrypted = manager.decrypt(encrypted)

        assert decrypted == plaintext

    def test_hash_api_key(self):
        """测试 API Key 哈希"""
        api_key = "sk-1234567890"
        hashed = SecurityManager.hash_api_key(api_key)

        assert len(hashed) == 16
        assert hashed != api_key

    def test_validate_api_key_valid(self):
        """测试有效 API Key 验证"""
        valid_keys = [
            "sk-1234567890abcdef",
            "Bearer token123456",
            "long_api_key_here_12345",
        ]

        for key in valid_keys:
            assert SecurityManager.validate_api_key(key) is True

    def test_validate_api_key_invalid(self):
        """测试无效 API Key 验证"""
        invalid_keys = [
            "",
            "short",
            "<your_api_key>",
            "your_key_here",
        ]

        for key in invalid_keys:
            assert SecurityManager.validate_api_key(key) is False

    def test_sanitize_input_basic(self):
        """测试基本输入清理"""
        manager = SecurityManager()
        assert manager.sanitize_input("  hello world  ") == "hello world"
        assert manager.sanitize_input("") == ""
        assert manager.sanitize_input(None) == ""

    def test_sanitize_input_xss(self):
        """测试 XSS 防护"""
        manager = SecurityManager()
        malicious_inputs = [
            '<script>alert("xss")</script>',
            "javascript:alert(1)",
            '<img onerror="alert(1)" src="x">',
            '<iframe src="evil.com"></iframe>',
        ]

        for input_text in malicious_inputs:
            sanitized = manager.sanitize_input(input_text)
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()

    def test_sanitize_input_length_limit(self):
        """测试长度限制"""
        manager = SecurityManager()
        long_input = "a" * 20000
        sanitized = manager.sanitize_input(long_input, max_length=10000)

        assert len(sanitized) <= 10000

    def test_validate_query_empty(self):
        """测试空查询验证"""
        manager = SecurityManager()
        is_valid, error_msg = manager.validate_query("")
        assert is_valid is False
        assert "不能为空" in error_msg

    def test_validate_query_too_long(self):
        """测试过长查询验证"""
        manager = SecurityManager()
        long_query = "a" * 6000
        is_valid, error_msg = manager.validate_query(long_query)
        assert is_valid is False
        assert "过长" in error_msg

    def test_validate_query_sql_injection(self):
        """测试 SQL 注入防护"""
        manager = SecurityManager()
        malicious_queries = [
            "DROP TABLE users",
            "DELETE FROM users WHERE 1=1",
            "INSERT INTO users VALUES (1, 'hacker')",
            "UPDATE users SET password='hacked'",
        ]

        for query in malicious_queries:
            is_valid, error_msg = manager.validate_query(query)
            assert is_valid is False
            assert "不被允许" in error_msg

    def test_validate_query_valid(self):
        """测试有效查询验证"""
        manager = SecurityManager()
        valid_queries = [
            "什么是 RAG？",
            "请解释深度学习原理",
            "今天天气怎么样？",
            "a" * 1000,  # 正常长度的查询
        ]

        for query in valid_queries:
            is_valid, error_msg = manager.validate_query(query)
            assert is_valid is True
            assert error_msg == ""


class TestSecurityManagerIntegration:
    """安全模块集成测试"""

    def test_full_encryption_workflow(self):
        """测试完整加密工作流"""
        manager = SecurityManager(secret_key="test_key")

        # 加密敏感数据
        api_key = "sk-secret-key-12345"
        encrypted = manager.encrypt(api_key)

        # 存储加密数据（模拟）
        stored_data = encrypted

        # 解密使用
        decrypted = manager.decrypt(stored_data)
        assert decrypted == api_key

    def test_sanitize_and_validate_chain(self):
        """测试清理和验证链式调用"""
        manager = SecurityManager()
        query = '  <script>alert("xss")</script> 什么是人工智能？  '

        # 先验证
        is_valid, _ = manager.validate_query(query)
        assert is_valid is True

        # 再清理
        sanitized = manager.sanitize_input(query)
        assert "<script>" not in sanitized
        assert sanitized == "什么是人工智能？"
