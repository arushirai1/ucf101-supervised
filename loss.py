import torch as th
import torch
import torch.nn.functional as F


def info_nce_loss(features, batch_size, views, temperature=0.05):
    '''
    Credit to Thalles Silva - https://github.com/sthalles/SimCLR/blob/master/simclr.py
    :param features:
    :param batch_size:
    :param views:
    :param temperature:
    :return:
    '''
    labels = torch.cat([torch.arange(batch_size) for i in range(views)], dim=0) #TODO: replace this with true labels for supervised contrastive learning
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool) #.to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0]) #.to(self.args.device)

    logits = logits / temperature
    return logits, labels
