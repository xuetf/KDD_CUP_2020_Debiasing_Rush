#!/usr/bin/env python3
# -*- coding:utf-8-*-
# 全局变量管理模块


def init():
    """在主模块初始化"""
    global GLOBALS_DICT
    GLOBALS_DICT = {}


def set_glv(name, value):
    """设置"""
    try:
        GLOBALS_DICT[name] = value
        return True
    except KeyError:
        return False


def get_glv(name):
    """取值"""
    try:
        return GLOBALS_DICT[name]
    except KeyError:
        return "Not Found"