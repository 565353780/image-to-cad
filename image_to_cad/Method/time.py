#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime

def getCurrentTimeStr():
    current_time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return current_time_str

