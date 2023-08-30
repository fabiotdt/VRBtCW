import os
import re
import numpy as np

# Get unique classes Calltech_24
calltech_categories = [re.sub(r'^\d+\.', '', item) for item in os.listdir('Calltech_24/')]

# create dictionary for imagenet classes
# key = class name, value = class id
with open("imagenet_classes.txt", "r") as f:
    imagenet_categories = f.readlines()
    imagenet_dict = {}
    for i in range(len(imagenet_categories)):
        imagenet_dict[imagenet_categories[i].strip().split(' ')[1]] = imagenet_categories[i].strip().split(' ')[0]
f.close()

# list of overlapping class names
common = list(set(calltech_categories).intersection(imagenet_dict.keys()))
# list of class names in imagenet but not in calltech
only_imagenet = list(set(imagenet_dict.keys()) - set(calltech_categories))
# list of overlapping class ids
id = [imagenet_dict[id] for id in common]
# list of class ids in imagenet but not in calltech
only_imagenet_id = [imagenet_dict[id] for id in only_imagenet]

print(len(common), ' classes in common')
print(len(only_imagenet), ' classes in imagenet but not in calltech', '\n')

# create a subset of only_imagenet of 20 classes
np.random.seed(0)
only_imagenet_subset = np.random.choice(only_imagenet, 20, replace=False)
only_imagenet_subset_id = [imagenet_dict[id] for id in only_imagenet_subset]

print('ONLY IMAGENET SUBSET', only_imagenet_subset)
print('ONLY IMAGENET SUBSET IDS', only_imagenet_subset_id)




