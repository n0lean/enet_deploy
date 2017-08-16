from __future__ import absolute_import
from utils.cityscapesscripts.helpers import labels
import pickle

all_labels = labels.labels
name = []
id = []
train_id = []
category = []
cat_id = []
has_instances = []
ignore_in_eval = []
color = []
for i in all_labels:
    name.append(i[0])
    id.append(i[1])
    train_id.append(i[2])
    category.append(i[3])
    cat_id.append(i[4])
    has_instances.append(i[5])
    ignore_in_eval.append(i[6])
    color.append(i[7])

dic = {
    'name': name,
    'id': id,
    'train_id': train_id,
    'category': category,
    'cat_id': cat_id,
    'has_instances': has_instances,
    'ignore_in_eval': ignore_in_eval,
    'color': color
}

pickle.dump(dic, file=open('dict.pkl','wb'))
