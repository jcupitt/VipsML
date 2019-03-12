# Install

```
virtualenv VipsML
source VipsML/bin/activate
pip install jupyter
jupyter notebook
pip install Keras pyvips 
```

We need a display driver, CUDA driver, cuda toolkit, and tendorflow-gpu 
package that all agree on version numbers. 

Ubuntu 18.10 ships with CUDA 9.1.

CUDA 10.0 is the version supported by tensorflow-gpu. The current version on
the nvidia site is CUDA 10.1 ... it is broken and no one uses it.

Go to:

https://developer.nvidia.com/cuda-zone

And download the runfile (local) installer of 10.0 from the archive for 18.04. 

```
sudo sh cuda_10.0.130_410.48_linux.run
```

You need to install everything. This won't work from the desktop, you have to
reboot, flip to a console session, and run there.



# Test

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

