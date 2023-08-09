import logging
from logging.handlers import TimedRotatingFileHandler



class Log():
    def set_log(self, filepath = "logs/log.log", level = 2, freq = "D", interval = 50, backup = 2, name = "log"):
        format = '%(asctime)s %(levelname)s %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        level_dict = {
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.ERROR,
            4: logging.WARNING,
            5: logging.CRITICAL,
        }

        fmt = logging.Formatter(format, datefmt)

        log_level = level_dict[level]

        self.logger = logging.getLogger(name = name)
        self.logger.setLevel(log_level)

        self.hdlr = TimedRotatingFileHandler(filename = filepath, when = freq, interval = interval, backupCount = backup, encoding='utf-8')
        self.hdlr.setFormatter(fmt)
        self.logger.addHandler(self.hdlr)

        return self.logger
    

    def shutdown(self):
        self.logger.removeHandler(self.hdlr)
        del self.logger, self.hdlr