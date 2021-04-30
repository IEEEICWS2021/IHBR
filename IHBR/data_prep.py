import torch.utils.data as data
import numpy as np

class CreateData(data.Dataset):
    def __init__(self, features, num_bundle, user_bundle_mat=None, num_neg=0, is_training=None):
        super(CreateData, self).__init__()

        self.features_pos = features
        self.features_neg = []
        self.features = []
        self.num_bundle = num_bundle
        self.user_bundle_mat = user_bundle_mat
        self.num_neg = num_neg
        self.is_training = is_training
        self.labels = [1] + [0 for _ in range(len(features)-1)]
        self.labels_all = []

    def neg_sample(self):
        self.features_neg = []
        for x in self.features_pos:
            user = x[0]
            for t in range(self.num_neg):
                b = np.random.randint(self.num_bundle)
                while (user, b) in self.user_bundle_mat:
                    b = np.random.randint(self.num_bundle)
                self.features_neg.append([user, b])

        labels_pos = [1 for _ in range(len(self.features_pos))]
        labels_neg = [0 for _ in range(len(self.features_neg))]
        self.labels_all = labels_pos + labels_neg
        self.features = self.features_pos + self.features_neg

    def __len__(self):
        return (self.num_neg + 1) * len(self.labels)

    def __getitem__(self, idx):
        if self.is_training:
            features = self.features
        else:
            features = self.features_pos
        if self.is_training:
            labels = self.labels_all
        else:
            labels = self.labels

        user = features[idx][0]
        bundle = features[idx][1]
        label = labels[idx]
        return user, bundle, label