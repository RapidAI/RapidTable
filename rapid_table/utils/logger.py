# -*- encoding: utf-8 -*-
# @Author: Jocker1212
# @Contact: xinyijianggo@gmail.com
import logging


class Logger:
    def __init__(self, log_level=logging.DEBUG, logger_name=None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        fmt = "[%(levelname)s] %(asctime)s RapidTable %(filename)s:%(lineno)d: %(message)s"
        formatter = logging.Formatter(fmt)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def get_log(self):
        return self.logger
