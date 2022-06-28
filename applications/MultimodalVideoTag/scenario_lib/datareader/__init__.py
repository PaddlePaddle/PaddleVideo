#!/usr/bin/env python
# coding=utf-8
"""
Copyright 2021 Baidu.com, Inc. All Rights Reserved
Description: 
Authors: wanghewei(wanghewei@baidu.com)
LastEditors: wanghewei(wanghewei@baidu.com)
Date: 2021-11-26 16:31:59
"""
from .reader_utils import regist_reader, get_reader
from .feature_reader import FeatureReader
# regist reader, sort by alphabet
regist_reader("ATTENTIONLSTMERNIE", FeatureReader)
