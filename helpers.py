import IPython.display
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from functools import reduce


def show_array(a, fmt='png'):
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def show_images(images):
    for i in range(images.shape[0]):
        if images.shape[1] <= 32:
            plt.figure()
            plt.imshow(images[i], interpolation='nearest')
        else:
            show_array(images[i])


def count_params(variables, scopes):
    results = []
    for pattern in scopes:
        count = sum([reduce(lambda a, b: a * b, v.get_shape().as_list(), 1)
                     for v in variables if pattern in v.name])
        results += ['%s: %.3fM' % (pattern, count / 1e6)]
    print(', '.join(results))
