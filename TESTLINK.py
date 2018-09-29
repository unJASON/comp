from distutils.version import LooseVersion
import warnings
import tensorflow as tf
# assert LooseVersion(tf.__version__)>=LooseVersion('1.0'),print("Tensorflow Version:{}".format(tf.__version__))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
if not tf.test.gpu_device_name():
    warnings.warn("not find GPU")
else:
    print("default GPU DEVICE:{}".format(tf.test.gpu_device_name))
