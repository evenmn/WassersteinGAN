import torch
import torch.nn as nn
import bz2
import _pickle as cPickle
import numpy as np

class SimplexDataset(torch.utils.data.Dataset):
    """Organize dataset as a subclass of the Dataset class. Then, minibatches can efficiently be
    loaded in and out of GPU memory
    """
    def __init__(self, root, isize=128):
        with bz2.BZ2File(root + 'simplex_friction_128x128.pbz2', 'rb') as f:
            dataset = cPickle.load(f)

        inputs = np.asarray(dataset['x'])
        print(inputs.shape)
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.view(-1, 1, isize, isize)
        self.data = nn.functional.interpolate(inputs, size=(isize, isize), mode='bilinear')
        print(self.data.shape)

        # features/targets/labels/attributes
        labels = np.asarray(dataset['y'])
        print(labels.shape)
        self.labels = torch.FloatTensor(labels)
        print(self.labels.shape)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y

    def __len__(self):
        return len(self.data)

