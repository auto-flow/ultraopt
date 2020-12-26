#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : qichun.tang@bupt.edu.cn
import socket

from ultraopt.utils.logging_ import get_logger

logger = get_logger(__name__)


def check_port_is_used(port, host="127.0.0.1") -> bool:
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    location = (host, port)
    result_of_check = a_socket.connect_ex(location)
    if result_of_check == 0:
        logger.warning(f"Port {location} is used.")
        return True
    else:
        logger.debug(f"Port {location} is not used.")

        return False

def get_a_free_port(start_port,host="127.0.0.1")->int:
    while check_port_is_used(start_port,host):
        start_port+=1
    return start_port