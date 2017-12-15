import os
import random

train = 0.5
test = 0.5
root = "/home/liuml/retinanet/images/VOC2007/ImageSets/Main"
input_file = "/home/liuml/retinanet/images/VOC2007/output.txt"
trainval_file = os.path.join(root, "trainval.txt")
test_file = os.path.join(root, "test.txt")
train_file = os.path.join(root, "train.txt")
val_file = os.path.join(root, "val.txt")

with open(input_file, 'r') as inf:
    lines = inf.readlines()
    random.shuffle(lines)

with open(trainval_file, 'w') as trivalf:
    for line in lines[0: int(len(lines) * train)]:
        trivalf.write(line[0:6] + '\n')

with open(train_file, 'w') as trif:
    for line in lines[0: int(len(lines) * train)]:
        trif.write(line[0:6] + '\n')

with open(test_file, 'w') as tesf:
    for line in lines[int(len(lines) * train): -1]:
        tesf.write(line[0:6] + '\n')

with open(val_file, 'w') as valf:
    for line in lines[int(len(lines) * train): -1]:
        valf.write(line[0:6] + '\n')
