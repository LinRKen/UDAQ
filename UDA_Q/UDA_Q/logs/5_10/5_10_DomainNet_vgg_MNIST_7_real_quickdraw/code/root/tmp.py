from configparser import ConfigParser
config = ConfigParser()
config.read('./UDA_Q/configs/MNIST_USPS.cfg', encoding='UTF-8')
a = config.items('optimizer')
print(a)