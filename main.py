from config import CHAR_EMBEDDING, FLAGS, DEBUG, MAX_WORDS, param, MAX_INGREDIENTS, MODEL_CHERCKPOINT
from dataset import datasets, read_ingredient, ingredients_clean, recipes_clean
from functools import partial
from time import time
import numpy as np
import os
import random
import tensorflow as tf
import traceback

#########
# Model #
#########


def char_conv(embedding, reuse, training):
    with tf.name_scope('char_conv'):
        kernel = 3
        net = tf.layers.conv1d(
            embedding, 45, kernel, name='conv1',
            reuse=reuse, padding='same')
        net = tf.layers.batch_normalization(
            net, training=training, name='bn1', reuse=reuse)
        net = tf.nn.relu(net)
        filters = 400
        net = tf.layers.conv1d(
            net, filters, kernel, name='conv2',
            reuse=reuse, padding='same')
        net = tf.layers.batch_normalization(
            net, training=training, name='bn2', reuse=reuse)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling1d(net, (net.shape[1],), 1)
        net = tf.reshape(net, [-1, filters])
        net = tf.layers.dense(net, 150, name='dense', reuse=reuse)
        net = tf.layers.batch_normalization(
            net, training=training, name='dense-bn', reuse=reuse)
        net = tf.nn.relu(net)
    return net


def model(ingredients, training):
    # Embed each ingredients
    net = char_embedding(ingredients, False)
    ingredients_conv = []
    with tf.name_scope('ingredients_conv'):
        for i, embedding in enumerate(tf.split(net, [1] * MAX_INGREDIENTS, 1)):
            embedding = tf.reshape(embedding, [-1, MAX_WORDS, CHAR_EMBEDDING])
            ingredients_conv += [char_conv(embedding, i != 0, training)]
    ingredients_conv = tf.stack(ingredients_conv, 1)
    ingredients_conv = tf.reduce_sum(ingredients_conv, 1)
    net = tf.layers.dense(ingredients_conv, 100, name='dense-last')
    net = tf.layers.batch_normalization(net, training=training, name='bn-last')
    tf.summary.histogram('recipe_vec', net)
    prediction = tf.layers.dense(net, len(param('cat2id')))
    return prediction, net


def char_embedding(x, reuse):
    with tf.variable_scope("char_embedding", reuse=reuse):
        initializer = None
        if not reuse:
            initializer = tf.contrib.layers.xavier_initializer()
            initializer = initializer([len(param('chars')), CHAR_EMBEDDING])
        params = tf.get_variable('embedding_lookup', initializer=initializer)
        net = tf.nn.embedding_lookup(params, tf.cast(x, tf.int32), )
    return net


def losses(predictions, labels, indexes, weights):
    for i in range(len(indexes)):
        indexes[i] = tf.constant(indexes[i])
        weights[i] = tf.constant(weights[i], tf.float32)

    # Root categories
    w_pred = predictions[:, indexes[0][0]:indexes[0][1]] * weights[0]
    h_loss = tf.losses.sparse_softmax_cross_entropy(
        labels[:, 0], w_pred, loss_collection=None,
        reduction=tf.losses.Reduction.NONE,)

    # Per line softmax
    def fn(i, depth, indexes, weights):
        def false_fn():
            w_pred = (predictions[i, indexes[i, 0]:indexes[i, 1]] *
                      weights[i][:indexes[i, 1] - indexes[i, 0]])
            return tf.losses.sparse_softmax_cross_entropy(
                labels=labels[i, depth],
                logits=w_pred,
                loss_collection=None)
        return tf.cond(tf.equal(labels[i, depth], -1),
                       lambda: 0.0,  # Make deep hierachies stronger
                       false_fn)

    # Explore hierarchical categories
    bs = FLAGS.batch_size
    for d in range(1, len(indexes)):
        h_loss += tf.map_fn(
            partial(
                fn,
                depth=d,
                indexes=tf.gather_nd(indexes[d], tf.maximum(0, labels[:, 0:d])),
                weights=tf.gather_nd(weights[d], tf.maximum(0, labels[:, 0:d]))),
            tf.range(bs),
            tf.float32)

    # Final loss
    h_loss = tf.reduce_mean(h_loss)
    tf.losses.add_loss(h_loss)
    loss = tf.losses.get_total_loss(True)
    tf.summary.scalar('loss', loss)
    return loss


############
# Training #
############

def train(sess):
    bs = FLAGS.batch_size
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.device('/cpu:0'):
        (title, categories, ingredients), iterator_inits = datasets(
            [([FLAGS.records_train], True),
             ([FLAGS.records_val], False), ])

    prediction, recipe_vec = model(ingredients, True)
    indexes = param('indexes')
    weights = param('weights')
    loss = losses(prediction, categories, indexes, weights)

    # Optimizer
    learning_rate = tf.train.polynomial_decay(
        FLAGS.learning_rate, global_step,
        decay_steps=40000, end_learning_rate=.0002)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            global_step=global_step)

    # Training loop
    sess.run(tf.global_variables_initializer())

    if 0:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_CHERCKPOINT)

    if not DEBUG:
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.logdir + '/val', sess.graph)

    if DEBUG:
        sess.run(iterator_inits[0])
        for i in range(2):
            start_time = time()
            try:
                _, loss_ = sess.run([
                    train_op, tf.losses.get_losses()])
            except tf.errors.InvalidArgumentError:
                traceback.print_exc()
                break
            duration = 1000 * (time() - start_time) / bs
            print('[%4s] Time %4dms, Loss %s' % (i, duration, loss_))
        return

    for epoch in range(0, 1000):
        # Training
        sess.run(iterator_inits[0])
        while True:
            try:
                _, summaries_, global_step_ = sess.run(
                    [train_op, summaries, global_step])
                train_writer.add_summary(summaries_, global_step_)
            except tf.errors.OutOfRangeError:
                break
        path = saver.save(sess, MODEL_CHERCKPOINT)
        print('Saving model', path)

        # Validation loss
        sess.run(iterator_inits[1])
        while True:
            try:
                _, summaries_ = sess.run([loss, summaries])
                val_writer.add_summary(summaries_, global_step_)
            except tf.errors.OutOfRangeError:
                break


#################
# Ingredient NN #
#################

def ingredient_nn(sess, query):
    bs = 64
    ingredients = tf.placeholder(tf.float32, shape=[None, MAX_WORDS])
    ingredient_a = tf.placeholder(tf.float32, shape=[None, 150])
    ingredient_b = tf.placeholder(tf.float32, shape=[None, 150])

    net = char_embedding(ingredients, False)
    ingredients_conv = char_conv(net, False, False)

    distance = tf.losses.cosine_distance(
        tf.nn.l2_normalize(ingredient_a, 1),
        tf.nn.l2_normalize(ingredient_b, 1), 1,
        reduction=tf.losses.Reduction.NONE)
    # distance = tf.losses.mean_squared_error(ingredient_a, ingredient_b,
    # reduction=tf.losses.Reduction.NONE) # Less good..

    # Init session
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, MODEL_CHERCKPOINT)

    # Grab ingredient vectors
    ingredients_np, ingredients_txt = get_ingredient_vecs(
        sess, ingredients_conv, ingredients)

    def get_single(query):
        needle = read_ingredient(query)
        needle = np.expand_dims(needle, 0)
        return sess.run(ingredients_conv, feed_dict={ingredients: needle})[0]

    results = []
    populations = random.sample(range(len(ingredients_txt)),
                                min(20000, len(ingredients_txt)) // 64 * 64)
    if not query:
        needles = np.tile(ingredients_np[populations[0]], (bs, 1))
    else:
        needles = np.tile(get_single(query), (bs, 1))
        results += [(0, query)]
    for i in range(0, len(populations), bs):
        indexes = populations[i:i + bs]
        feed_dict = {ingredient_a: needles,
                     ingredient_b: ingredients_np[indexes]}
        distances = sess.run(distance, feed_dict=feed_dict)
        for j, ing_index in enumerate(indexes):
            results += [(distances[j][0], ingredients_txt[ing_index], )]
    return sorted(results, key=lambda x: x[0])


def get_ingredient_vecs(sess, ingredients_conv, ingredients):
    path = '/tmp/ingredients.np.npy'
    ingredients_txt = ingredients_clean()
    if os.path.exists(path):
        return np.load(path), ingredients_txt
    bs = 1024
    results = []
    for i in range(0, len(ingredients_txt), bs):
        batch = np.zeros([bs, MAX_WORDS])
        for j, txt in enumerate(ingredients_txt[i:i + bs]):
            batch[j, :] = read_ingredient(txt)
        ingredients_np = sess.run(ingredients_conv, feed_dict={ingredients: batch})
        results += [ingredients_np]
        print(i)
    ingredients_np = np.concatenate(results)
    ingredients_np = np.squeeze(ingredients_np)
    np.save(path, ingredients_np)
    return ingredients_np, ingredients_txt


#######
# Run #
#######


def main(_):
    with tf.Session() as sess:
        if FLAGS.inference:
            results = ingredient_nn(sess, None)
            data = results[:20]
            print('{:-^80}'.format(data[0][1]))
            for x in data[1:]:
                print("%.4f: %s" % x)
        else:
            train(sess)
    return


if __name__ == '__main__':
    tf.app.run()
