# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from urllib.parse import urlparse


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
