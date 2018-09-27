# Hierarchical categories loss (Tensorflow)

A loss function that takes into account categories with a hierarchical structure.

This project is an attempt to learn a cooking recipe embedding from ingredients (at a character level). The model loss function is learning from hierarchical categories. For example, your target labels can be "Breakfast/waffles" or "Poultry/Turkey/Ground".

## Requirements

  * Python 3+
  * `pip install -r requirements.txt`

## Getting started

Building the TFRecords

`python dataset.py --data_dir sample/ --records_val /tmp/val.recs --records_train /tmp/train.recs`

Training the model:

`python main.py --records_val /tmp/val.recs --records_train /tmp/train.recs --logdir /tmp/experiment`
