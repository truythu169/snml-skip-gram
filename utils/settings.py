import configparser
import os

# root directory path
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# load configs
config = configparser.ConfigParser()
config.read(root_dir + '/config.ini', encoding='utf-8')

# load environments parameters
env = configparser.ConfigParser()
env.read(root_dir + '/env.ini', encoding='utf-8')
