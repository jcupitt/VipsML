# Install

Version requirements:

* `tensorflow-gpu==0.12` needs `libcudart.so.8.0`
* `tensorflow-gpu==1.1` needs `libcublas.so.8.0`
* `tensorflow-gpu==1.12` needs `libcublas.so.9.0`
* `tensorflow-gpu==1.13` needs `libcublas.so.10.0`.

Ubuntu 18.10 has 9.1, so you need to build your own `tensorflow-gpu` from
source.

Imperial Ubuntu has `libcuda.so.384.130`, but nothing else. Again, build your
own `tensorflow-gpu`.

```
virtualenv -p $(which python3) VipsML
source VipsML/bin/activate
pip install jupyter
pip install Keras pyvips 
pip install tensorflow
jupyter notebook
```

# Test for gpu tensorflow

```python
import tensorflow as tf
print(“tf version = “, tf.__version__)
with tf.device(‘/gpu:0’):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name=’a’)
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name=’b’)
  c = tf.matmul(a, b)

with tf.Session() as sess:
  print (sess.run(c))
```

