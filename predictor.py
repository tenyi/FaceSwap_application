"""Modernised PRNet ResFCN-256 predictor (TensorFlow 2.x, graph mode via compat.v1).

The original `predictor.py` used `tensorflow.contrib.layers`, which was removed in
TensorFlow 2.0.  This module rebuilds the *same* graph -- identical variable
names and shapes, so the released `256_256_resfcn256_weight` checkpoint still
restores -- using only `tf.compat.v1` primitives.

contrib semantics reproduced here (verified against the checkpoint contents):
  * conv scope default name  : 'Conv' / 'Conv2d_transpose'
  * kernel variable name     : 'weights'  (contrib renames v1-layer 'kernel')
  * bias variable name       : 'biases'   (never created: use_bias is False)
  * batch-norm scope         : 'BatchNorm' (nested inside the conv scope)
  * batch-norm variables     : gamma / beta / moving_mean / moving_variance
  * batch-norm defaults      : center=True, scale from arg_scope, epsilon=0.001
  * ordering                 : conv -> batch-norm -> activation
"""
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# contrib.layers._build_variable_getter({'bias': 'biases', 'kernel': 'weights'})
_RENAME = {'bias': 'biases', 'kernel': 'weights'}

# TF 2.x legacy (Keras-backed) v1 layers open an extra scope named after the
# layer class ("conv2d" / "conv2d_transpose") that contrib did not have.
# contrib passed `name=sc.name` + `_scope=sc` to avoid it; we strip it instead.
_LAYER_SCOPES = ('conv2d_transpose', 'conv2d')

# contrib.layers.batch_norm default epsilon (TF 1.14).
_BN_EPSILON = 0.001


def _contrib_variable_getter(getter, name, registered=False, **kwargs):
    """contrib.layers._build_variable_getter: kernel->weights, bias->biases."""
    for old_name, new_name in _RENAME.items():
        if name.endswith(old_name):
            name = name[:-len(old_name)] + new_name
            break
    parts = name.split('/')
    if len(parts) > 1 and parts[-2] in _LAYER_SCOPES:
        parts = parts[:-2] + [parts[-1]]
        name = '/'.join(parts)
    kwargs.pop('registered', None)
    return getter(name, **kwargs)


def batch_norm(inputs, is_training=False, scale=True, center=True,
               epsilon=_BN_EPSILON, scope=None):
    """Drop-in replacement for contrib.layers.batch_norm (inference mode)."""
    with tf.variable_scope(scope, default_name='BatchNorm', values=[inputs]):
        inputs = tf.convert_to_tensor(inputs)
        params_shape = inputs.get_shape().as_list()[-1:]
        beta = tf.get_variable('beta', params_shape,
                               initializer=tf.zeros_initializer(), dtype=inputs.dtype)
        if scale:
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer(), dtype=inputs.dtype)
        else:
            gamma = None
        moving_mean = tf.get_variable('moving_mean', params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False, dtype=inputs.dtype)
        moving_variance = tf.get_variable('moving_variance', params_shape,
                                          initializer=tf.ones_initializer(),
                                          trainable=False, dtype=inputs.dtype)

    # Inference: use the moving statistics stored in the checkpoint.
    return tf.nn.batch_normalization(inputs, moving_mean, moving_variance,
                                     beta, gamma, epsilon)


def conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME',
           activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
           biases_initializer=None, scope=None):
    """Drop-in replacement for contrib.layers.conv2d.

    Defaults mirror the original `arg_scope([tcl.conv2d, ...])` block, i.e.
    every convolution is followed by a batch-norm and a ReLU, and never owns a
    bias term.
    """
    with tf.variable_scope(scope, default_name='Conv', values=[inputs],
                           custom_getter=_contrib_variable_getter):
        layer = tf.layers.Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=None,
            use_bias=not normalizer_fn and biases_initializer,
            name=None,
        )
        outputs = layer.apply(inputs)
        if normalizer_fn is not None:
            outputs = normalizer_fn(outputs)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs, num_outputs, kernel_size, stride=1, padding='SAME',
                     activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
                     biases_initializer=None, scope=None):
    """Drop-in replacement for contrib.layers.conv2d_transpose."""
    with tf.variable_scope(scope, default_name='Conv2d_transpose', values=[inputs],
                           custom_getter=_contrib_variable_getter):
        layer = tf.layers.Conv2DTranspose(
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=None,
            use_bias=not normalizer_fn and biases_initializer,
            name=None,
        )
        outputs = layer.apply(inputs)
        if normalizer_fn is not None:
            outputs = normalizer_fn(outputs)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu,
             normalizer_fn=batch_norm, scope=None):
    """Same topology as the original predictor.resBlock.

    The original relied on `arg_scope` to inject `normalizer_fn=tcl.batch_norm`
    into every convolution; here the same function object is forwarded
    explicitly.
    """
    assert num_outputs % 2 == 0
    with tf.variable_scope(scope, default_name='resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                               activation_fn=None, normalizer_fn=None,
                               biases_initializer=None, scope='shortcut')
        x = conv2d(x, num_outputs // 2, kernel_size=1, stride=1, padding='SAME',
                   normalizer_fn=normalizer_fn)
        x = conv2d(x, num_outputs // 2, kernel_size=kernel_size, stride=stride,
                   padding='SAME', normalizer_fn=normalizer_fn)
        x = conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None,
                   normalizer_fn=None, biases_initializer=None)

        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3,
                 name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with tf.variable_scope(self.name):
            # mirrors `arg_scope([tcl.batch_norm], is_training=..., scale=True)`
            bn = lambda t: batch_norm(t, is_training=is_training, scale=True)
            size = 16
            se = conv2d(x, num_outputs=size, kernel_size=4, stride=1,
                        normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2,
                          normalizer_fn=bn)
            se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1,
                          normalizer_fn=bn)

            pd = conv2d_transpose(se, size * 32, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 16, 4, stride=2, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 16, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 16, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 8, 4, stride=2, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 8, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 8, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 4, 4, stride=2, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 4, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 4, 4, stride=1, normalizer_fn=bn)

            pd = conv2d_transpose(pd, size * 2, 4, stride=2, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size * 2, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size, 4, stride=2, normalizer_fn=bn)
            pd = conv2d_transpose(pd, size, 4, stride=1, normalizer_fn=bn)

            pd = conv2d_transpose(pd, 3, 4, stride=1, normalizer_fn=bn)
            pd = conv2d_transpose(pd, 3, 4, stride=1, normalizer_fn=bn)
            pos = conv2d_transpose(pd, 3, 4, stride=1, normalizer_fn=bn,
                                    activation_fn=tf.nn.sigmoid)
            return pos

    @property
    def vars(self):
        return [var for var in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) if self.name in var.name]


class PosPrediction():
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # Each predictor owns a private graph so that several instances can
        # coexist in one process (the original code reused the default graph
        # and crashed with "Variable resfcn256/Conv/weights already exists"
        # as soon as a second instance was created).
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.network = resfcn256(self.resolution_inp, self.resolution_op)

            self.x = tf.placeholder(
                tf.float32,
                shape=[None, self.resolution_inp, self.resolution_inp, 3])
            self.x_op = self.network(self.x, is_training=False)

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config, graph=self._graph)
            self._saver = tf.compat.v1.train.Saver(self.network.vars)

    def restore(self, model_path):
        with self._graph.as_default():
            self._saver.restore(self.sess, model_path)

    def predict(self, image):
        pos = self.sess.run(self.x_op, feed_dict={self.x: image[np.newaxis, :, :, :]})
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.MaxPos
