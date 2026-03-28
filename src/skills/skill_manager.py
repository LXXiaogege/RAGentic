# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class SkillMeta:
    name: str
    description: str
    file_path: str


class SkillManager:
    def __init__(self, skills_dir: str):
        if not os.path.isabs(skills_dir):
            # 相对路径解析为项目根目录下
            project_root = Path(__file__).resolve().parents[2]
            skills_dir = str(project_root / skills_dir)
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, SkillMeta] = {}
        self._load_skills()

    def _load_skills(self):
        """扫描 skills 目录，解析所有 .md 文件的 frontmatter"""
        if not self.skills_dir.exists():
            logger.debug(f"skills 目录不存在：{self.skills_dir}")
            return

        for md_file in self.skills_dir.glob("*.md"):
            meta = self._parse_frontmatter(md_file)
            if meta:
                self._skills[meta.name] = meta
                logger.debug(f"加载 skill：{meta.name} - {meta.description}")

        logger.info(f"共加载 {len(self._skills)} 个 skills：{list(self._skills.keys())}")

    def _parse_frontmatter(self, file_path: Path) -> Optional[SkillMeta]:
        """解析 md 文件的 YAML frontmatter，提取 name 和 description"""
        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.startswith("---"):
                return None

            end = content.find("---", 3)
            if end == -1:
                return None

            frontmatter = content[3:end].strip()
            name = None
            description = None

            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if key == "name":
                        name = value
                    elif key == "description":
                        description = value

            if not name or not description:
                logger.warning(f"skill 文件缺少 name 或 description：{file_path}")
                return None

            return SkillMeta(name=name, description=description, file_path=str(file_path))

        except Exception as e:
            logger.warning(f"解析 skill 文件失败 {file_path}：{e}")
            return None

    def get_skills_prompt_block(self) -> str:
        """生成注入系统提示词的 <available_skills> 块，无 skill 时返回空字符串"""
        if not self._skills:
            return ""

        lines = [
            "<available_skills>",
            "你可以使用以下 Skills 来更好地完成特定任务。当用户请求匹配某个 skill 时，",
            "使用 read_skill 工具读取该 skill 的完整指令，然后严格按照指令执行。",
            "",
        ]
        for meta in self._skills.values():
            lines.append(f"- {meta.name}: {meta.description}")
        lines.append("</available_skills>")

        return "\n".join(lines)

    def get_skill_body(self, name: str) -> Optional[str]:
        """读取指定 skill 的 md 正文内容（frontmatter 之后的部分）"""
        meta = self._skills.get(name)
        if not meta:
            return None

        try:
            content = Path(meta.file_path).read_text(encoding="utf-8")
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    return content[end + 3:].strip()
            return content.strip()
        except Exception as e:
            logger.warning(f"读取 skill 正文失败 {name}：{e}")
            return None

    @property
    def skills(self) -> Dict[str, SkillMeta]:
        return self._skills
