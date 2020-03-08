# 2020/02/29 by ausk

import warnings
warnings.filterwarnings("ignore")

import os, sys, glob
import contextlib
import time

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.keras import models
import tensorflow.keras.backend as K

#tf.compat.v1.disable_eager_execution()

#assert tf.__version__[:4] == "1.4.", "Tensorflow should be v1.4.x!"

# Reset the GPU session
def reset_gpu_session(devices="0", fraction=0.8):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.log_device_placement = False
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras
    return sess

# To get the version.
def get_version(version):
    assert isinstance(version, str)
    import re
    return list(map(int, re.findall("(\d+)", version)))[:3]

# To make sure the directory exist.
def mksured(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath, exist_ok=True)

# The contextmanager to avoid raise except.
@contextlib.contextmanager
def warnOnException():
    try:
        yield
    except Exception as exc:
        print("Exception: {}".format(exc))
    finally:
        pass

# The contextmanager to time the code.
@contextlib.contextmanager
def timeit():
    ts = time.time()
    try:
        yield
    finally:
        pass
    te = time.time()
    print("dt: {:.3f} ms ".format( (te-ts)*1000) )


# Save the graph in current session as binary protobuf file.
def savepb(model, sess=None, pbname="result.pb"):
    #tf.keras.backend.set_learning_phase(0)
    #K.set_learning_phase(0)
    model_outputs = model.outputs
    output_names = []
    outputs = []
    for i, output in enumerate(model_outputs):
        output_names.append("output_{}".format(i))
        outputs.append(tf.identity(output, name = output_names[i]))

    input_names = [node.op.name for node in model.inputs]

    print("[ INFO ] inputs: ", input_names)
    print("[ INFO ] outputs: ", output_names)

    if sess is None:
        sess = K.get_session()

    graph_def = sess.graph.as_graph_def()
    #graphdef_inf = graph_util.remove_training_nodes(graph_def)
    #graphdef_frozen = graph_util.convert_variables_to_constants(sess, graphdef_inf, output_names)
    graphdef_frozen = graph_util.convert_variables_to_constants(sess, graph_def, output_names)
    graph_io.write_graph(graphdef_frozen, ".", pbname, as_text=False)
    print("[ INFO ] Saved pb into: ", pbname)
    print("[ INFO ] inputs: ", input_names)
    print("[ INFO ] outputs: ", output_names)


def loadGraph(pbfpath, prefix=""):
    """
    (1) About prefix.
    If we set name(prefix) = None，then it will add "import/" prefix to each node.
    And we should use name(prefix) = "" to keep none prefix.
    (2) It may be faild when load pb graph if it contains BN layers.
    """
    tf.keras.backend.set_learning_phase(0)
    K.set_learning_phase(0)
    K.clear_session()

    # (1) Load the protobuf file from the disk and parse it to retrieve the unserialized graph_def.
    #with tf.gfile.GFile(pbfpath, "rb") as fin:
    with open(pbfpath, "rb") as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())

     # (2) Import a graph_def into the current default Graph.
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=prefix,
            op_dict=None,
            producer_op_list=None
        )

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        pass
        #print(op.name)

    return graph
    """
    # Access the input and output nodes
    x = graph.get_tensor_by_name(prefix+'input_1:0')
    y = graph.get_tensor_by_name(prefix+'output_0:0')

    # Launch a Session and load the graph, feed and interence.
    with tf.Session(graph=graph) as sess:
        xinput = np.random.random((1, 256, 256,1))
        pred_y = sess.run( y, feed_dict={x: xinput} )
        print(pred_y)
    """

def keras2tflite(h5fpath, litefpath=None):
    if litefpath is None:
        litefpath = h5fpath[:-2] + "tflite"
    model = models.load_model(h5fpath)
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    data = c.convert()
    with open(litefpath, "wb") as fout:
        fout.write(data)