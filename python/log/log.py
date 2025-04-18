import logging
import os
from datetime import datetime

# 记录器
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
# setLevel注意：
# logger默认为warning。StreamHandler和FileHandler如果比warning低，就不会显示，此时需要重新设置logger。
# logger和StreamHandler、FileHandler有冲突，以logger为主。如果想分别设置，就设置logger为两者中更低的。
# 在setLevel时，StreamHandler和FileHandler分别设置，然后logger设置为两者中更低的，三者均需要用户设置。

# 处理器
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
log_filename = os.path.join(log_dir, f'{timestamp}.log')
fileHandler = logging.FileHandler(filename=log_filename)
fileHandler.setLevel(logging.INFO)

# formatter格式
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)8s | %(message)s')

# 给处理器设置格式
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# 记录器要设置处理器
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

logger.debug('debug message')
logger.info('info message')
logger.warning('warning message')
logger.error('error message')
logger.critical('critical message')
