import sys
import logging
from pathlib import Path

def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    log_format=None
):
    """
    configure the logging system for the project
    
    parameters:
        log_level: log level, default INFO
        log_file: log file path, if None then only output to console
        log_format: log format string, if None then use default format
    """
    # default log format
    if log_format is None:
        log_format = '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
    
    # create formatter
    formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # clear existing handlers (avoid duplicate addition)
    root_logger.handlers.clear()
    
    # create console handler (output to console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # if log file is specified, create file handler
    if log_file:
        # ensure log file directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # file record all levels
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# ========== 使用示例 ==========
if __name__ == "__main__":
    # method 1: simple configuration (only output to console)
    # setup_logging(log_level=logging.INFO)
    
    # method 2: configure log file
    setup_logging(
        log_level=logging.INFO,
        log_file="logs/test.log"
    )
    
    # method 3: use DEBUG level when developing
    # setup_logging(log_level=logging.DEBUG)
    
    # now logging can be used in any module
    logger = logging.getLogger(__name__)
    # logger.info("日志系统配置完成")
    # logger.debug("这是调试信息")
    # logger.warning("这是警告信息")
    # logger.error("这是错误信息")


