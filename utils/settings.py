import configparser
import os

# root directory path
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# load configs
config = configparser.ConfigParser()
config.read(root_dir + '/config.ini', encoding='utf-8')
