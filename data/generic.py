# -*- coding: utf-8 -*-

"""
    General data settings
"""

from enum import Enum, unique

@unique
class Role(str, Enum):
    USER = "USER"
    ASSISTANT = "BOT"

@unique
class NPCType(str, Enum):
    REAL = "真实人物"  # Real person
    VIRTUAL = "虚拟IP"  # Virtual IP
    DAILY = "日常人物"  # Daily person
    COMPANION = "情感陪伴"  # Emotional companion
