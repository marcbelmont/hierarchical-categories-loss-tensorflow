from collections import OrderedDict
from config import FLAGS, MAX_WORDS, MAX_INGREDIENTS, param, MAX_TITLE
from glob import glob
import json
import jsonlines
import numpy as np
import os
import re, unidecode
import tensorflow as tf

############
# TFRecord #
############


def convert_to_record(recipes, filename):
    def int64_f(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def float_f(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_f(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    cat2id = param('cat2id')
    depth = param('depth')
    _, table = char_tables(param('chars'))

    # Write records
    print('Creating TFRecords', filename, len(recipes), 'elements')
    writer = tf.python_io.TFRecordWriter(filename)
    for i, recipe in enumerate(recipes):
        if i % 500 == 0:
            print(i)

        categories = np.zeros(depth, np.int64) - 1
        hierarchy = (recipe['hierarchy'] + ['Other'])[:depth]
        for j in range(0, len(hierarchy)):
            key = '/'.join(hierarchy[:j + 1])
            if key not in cat2id:
                key = '/'.join(hierarchy[:j] + ['Other'])
            if key not in cat2id:
                break
            categories[j] = cat2id[key]

        if categories[0] == -1:
            continue

        # Labels
        title = clean_text(recipe['title'])
        title = bytes(title[:MAX_TITLE].translate(table), 'ascii')
        feature = dict(categories=int64_f(categories),
                       title=bytes_f([title]),
                       ingredients=int64_f([len(recipe['ingredients'])]))

        # Ingredients
        for i, ingredient in enumerate(recipe['ingredients']):
            if i > MAX_INGREDIENTS:
                break
            ingredient = clean_text(ingredient)
            ingredient = bytes(ingredient[:MAX_WORDS].translate(table), 'ascii')
            feature['ingredient%i' % i] = bytes_f([ingredient])
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

###########
# Dataset #
###########


def create_dataset(records, shuffle):
    max_categories = param('depth')
    default = tf.constant(chr(0))

    def parse_function(serialized):
        features = dict(
            categories=tf.FixedLenFeature([max_categories], tf.int64),
            title=tf.FixedLenFeature([], tf.string))
        for i in range(MAX_INGREDIENTS):
            features['ingredient%d' % i] = tf.FixedLenFeature(
                [], tf.string, default)
        parsed = tf.parse_single_example(serialized, features=features)
        categories = tf.cast(parsed['categories'], tf.int32)
        title = tf.decode_raw(parsed['title'], tf.uint8)
        title = tf.pad(title, [[0, MAX_TITLE - tf.shape(title)[0]]])
        title.set_shape([MAX_TITLE])

        # Ingredients
        ingredients = []
        for i in range(MAX_INGREDIENTS):
            ingredient = tf.decode_raw(parsed['ingredient%d' % i], tf.uint8)
            ingredient = tf.pad(ingredient, [[0, MAX_WORDS - tf.shape(ingredient)[0]]])
            ingredients += [ingredient]
        ingredients = tf.stack(ingredients)
        ingredients.set_shape([MAX_INGREDIENTS, MAX_WORDS])
        return title, categories, ingredients
    dataset = tf.data.TFRecordDataset(records)
    dataset = dataset.map(parse_function, num_parallel_calls=16)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.filter(
        lambda x, *_: tf.equal(tf.shape(x)[0], FLAGS.batch_size))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(10000)
    return dataset


def datasets(dataset_list):
    result = []
    for records, shuffle in dataset_list:
        result += [create_dataset(records, shuffle)]
    iterator = tf.data.Iterator.from_structure(
        result[0].output_types, result[0].output_shapes)
    next_element = iterator.get_next()
    return (next_element,
            [iterator.make_initializer(x) for x in result])

##############
# Allrecipes #
##############


def allrecipe_tree(recipes):
    tree = OrderedDict()

    def add_node(hierarchy, t):
        key, *rest = hierarchy
        if key not in t:
            t[key] = (0, OrderedDict())
        t[key] = (t[key][0] + 1,
                  add_node(rest, t[key][1]) if rest else t[key][1])
        return t
    for recipe in recipes:
        add_node(recipe['hierarchy'], tree)

    # strip subcategories that are too small
    def strip(tree):
        for key in list(tree.keys()):
            if tree[key][0] < 40:
                del tree[key]
        for key in tree.keys():
            tree[key] = (tree[key][0], strip(tree[key][1]))
        return tree
    tree = strip(tree)

    def fix_missing(tree):
        for key in list(tree.keys()):
            size, children = tree[key]
            children_size = sum([x[0] for x in children.values()])
            if size >= children_size and children_size > 0:
                size_other = size - children_size
                if size_other == 0 and len(children.values()) == 1:
                    del children[list(children.keys())[0]]
                elif size_other > 0:
                    children['Other'] = (size_other, {})
        for key in tree.keys():
            tree[key] = (tree[key][0], fix_missing(tree[key][1]))
        return tree
    tree = fix_missing(tree)

    def as_list(tree):
        res = []
        total = 0
        for key, (count, subtree) in tree.items():
            total += count
        for key, (count, subtree) in tree.items():
            res += [(key, count / total, as_list(subtree))]
        return res

    return as_list(tree)


def count_nodes(tree):
    count = 0
    for _, __, node in tree:
        count += count_nodes(node) + 1
    return count


def tree_vecs(tree):
    def max_siblings(tree, depth):
        if depth == 0:
            return len(tree)
        results = []
        for key, count, subtree in tree:
            results += [max_siblings(subtree, depth - 1)]
        return max(results) if results else 0

    def tree_index(tree, coord):
        index, *coord = coord
        if index >= len(tree):
            return None
        key, count, subtree = tree[index]
        if not coord:
            return subtree
        return tree_index(subtree, coord)

    start, end = 0, max_siblings(tree, 0)
    weights = [[x[1] for x in tree]]
    indexes = [[start, end]]
    start = end
    l1 = np.zeros([max_siblings(tree, 0), 2], np.int)
    l1_weights = np.zeros([max_siblings(tree, i) for i in range(2)])
    for i in range(l1.shape[0]):
        node = tree_index(tree, [i])
        if node:
            end = start + len(node)
            l1_weights[i][0:len(node)] = [x[1] for x in node]
            l1[i] = [start, end]
            start = end
    indexes += [l1.tolist()]
    weights += [l1_weights.tolist()]

    l2 = np.zeros([max_siblings(tree, 0), max_siblings(tree, 1), 2], np.int)
    l2_weights = np.zeros([max_siblings(tree, i) for i in range(3)])
    for i in range(l2.shape[0]):
        for j in range(l2.shape[1]):
            node = tree_index(tree, [i, j])
            if node:
                end = start + len(node)
                l2_weights[i, j][0:len(node)] = [x[1] for x in node]
                l2[i, j] = [start, end]
                start = end
    indexes += [l2.tolist()]
    weights += [l2_weights.tolist()]

    cat2id = {}

    def walk_bf(tree, prefix):
        for i, (key, count, subtree) in enumerate(tree):
            cat2id[prefix + key] = i
            walk_bf(subtree, prefix + key + '/')
    walk_bf(tree, '')

    return dict(indexes=indexes,
                cat2id=cat2id,
                depth=3,
                weights=weights,
                tree=tree)


def save_allrecipes():
    recipes = allrecipes()
    tree = allrecipe_tree(recipes)
    result = tree_vecs(tree)
    result.update(charset(recipes))
    with open(os.path.join(FLAGS.data_dir, 'allrecipes-info.json'), 'w') as f:
        json.dump(result, f)
    return result


def allrecipes():
    filenames = glob(os.path.join(FLAGS.data_dir, 'recipes*.jl'))
    recipes = []
    dups = set()
    dups_count = 0
    blacklist = ['Everyday Cooking', 'World Cuisine', 'U.S. Recipes',
                 'Events and Gatherings', 'Trusted Brands: Recipes and Tips']
    total = 0

    for filename in filenames:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                total += 1
                # Skip dups
                title = obj.get('title')[0]
                if title in dups:
                    dups_count += 1
                    continue
                dups.add(title)

                # Save recipe
                hierarchy = obj.get('hierarchy')[2:]
                if (not hierarchy or  # Different behaviour on server?!
                    any(x == hierarchy[0] for x in blacklist) or
                    hierarchy == ['Main Dish']):
                    continue
                ingredients = obj.get('ingredients')[:MAX_INGREDIENTS]
                ingredients = [clean_text(x)[:MAX_WORDS] for x in ingredients]
                recipes += [dict(
                    title=title, hierarchy=hierarchy, ingredients=ingredients)]
    print('Total', total, 'Duplicates', dups_count, 'Valid', len(recipes))
    with open(os.path.join(FLAGS.data_dir, 'allrecipes.json'), 'w') as f:
        json.dump(recipes, f)
    return recipes


###########
# Helpers #
###########


def read_ingredient(ingredient):
    _, table = char_tables(param('chars'))
    ingredient = clean_text(ingredient)
    ingredient = bytes(ingredient[:MAX_WORDS].translate(table), 'ascii')
    ingredient = np.frombuffer(ingredient, dtype=np.uint8)
    ingredient = np.pad(ingredient, [0, MAX_WORDS - ingredient.shape[0]], 'constant')
    return ingredient


def recipes_clean():
    with open(os.path.join(FLAGS.data_dir, 'allrecipes.json'), ) as f:
        recipes = json.load(f)

    def f(item):  # hack to align this list with tfrecords
        if item['hierarchy'][0] in ['Holidays and Events', 'Fruits and Vegetables', 'Holidays and Events']:
            return False
        return True
    return list(filter(f, recipes))


def ingredients_clean():
    recipes = recipes_clean()
    ingredients_txt = set()
    for recipe in recipes:
        ingredients_txt |= set([clean_text(x) for x in recipe['ingredients']])
    return sorted(list(ingredients_txt))


def charset(recipes):
    chars = set()
    for recipe in recipes:
        ingredients = clean_text(''.join(recipe['ingredients']))
        ingredients += clean_text(recipe['title'])
        chars |= set(ingredients)
    chars = ''.join(sorted(list(chars)))
    return dict(chars=chars)


def char_tables(chars):
    id2char = str.maketrans(''.join([chr(i) for i in range(len(chars))]), chars, )
    char2id = str.maketrans(chars, ''.join([chr(i) for i in range(len(chars))]))
    return id2char, char2id


regex_punctuation = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~/'))
regex_space = re.compile('\s+')
regex_numbers = re.compile('[0-9]')
regex_words = re.compile('cups?|teaspoons?|tablespoons?|pounds?')


def clean_text(text):
    text = unidecode.unidecode(text).lower()
    text = regex_punctuation.sub(' ', text)
    text = regex_words.sub(' ', text)
    text = regex_space.sub(' ', text)
    return text


def main():
    save_allrecipes()
    recipes = recipes_clean()
    if not recipes:
        print('No recipes found!')
        return
    convert_to_record(recipes[:1000], FLAGS.records_val)
    convert_to_record(recipes[1000:], FLAGS.records_train)


if __name__ == '__main__':
    main()
