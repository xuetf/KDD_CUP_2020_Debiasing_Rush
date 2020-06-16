#!/usr/bin/env python3
# -*- coding:utf-8-*-
# global variable management


def init():
    global GLOBALS_DICT
    GLOBALS_DICT = {}


def set_glv(name, value):
    try:
        GLOBALS_DICT[name] = value
        return True
    except KeyError:
        return False


def get_glv(name):
    try:
        return GLOBALS_DICT[name]
    except KeyError:
        return "Not Found"
