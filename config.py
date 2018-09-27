from subprocess import check_output
import json
import os
import tensorflow as tf
from functools import lru_cache

tf.flags.DEFINE_string('logdir', None, 'Log directory')
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_float("learning_rate", .0005, "Learning rate")
tf.flags.DEFINE_string('records_val', 'cache/allrecipes-val.tfrecords', 'TFrecord files')
tf.flags.DEFINE_string('records_train', 'cache/allrecipes-train.tfrecords', 'TFrecord files')
tf.flags.DEFINE_string('data_dir', 'sample', 'Directory with json data')
tf.flags.DEFINE_boolean('inference', False, 'Find nearest neighbor for ingredients')

FLAGS = tf.flags.FLAGS
DEBUG = False

MAX_INGREDIENTS = 24
MAX_WORDS = 100
MAX_TITLE = 70
CHAR_EMBEDDING = 15  # https://github.com/carpedm20/lstm-char-cnn-tensorflow
ERROR_WEIGHTS = dict(calories=.3, fat=.5, protein=.5, sodium=.3)
ERROR_WEIGHTS = dict()
try:
    MODEL_CHERCKPOINT = "save/%s/model.ckpt" % check_output(
        ['git', 'describe', '--abbrev=0', '--tags']).strip().decode('ascii')
except:
    MODEL_CHERCKPOINT = "save/model.ckpt"


@lru_cache(maxsize=32)
def param(name):
    path = os.path.join(FLAGS.data_dir, 'allrecipes-info.json')
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        results = json.load(fp)
    return results[name]
