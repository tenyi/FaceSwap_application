import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

# The released `seg_model_384.pb` is a frozen TensorFlow 1.x graph.
# TF2 removed the v1 session API, so we drive the graph through compat.v1.
tf.disable_eager_execution()

curPath = os.path.abspath(os.path.dirname(__file__))


class FaceSeg:
    def __init__(self, model_path=os.path.join(curPath, 'seg_model_384.pb')):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        with self._graph.as_default():
            graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        self._sess = tf.compat.v1.Session(config=config, graph=self._graph)

        self.pb_file_path = model_path
        self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
        self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image_input = (image / 255.)[np.newaxis, :, :, :]
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]
        return self.output_transform(output, shape=image.shape[:2])
