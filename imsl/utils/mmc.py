import torch
import numpy as np

def relabel_labels(labels):
    unique_labels = np.unique(labels)
    #print(unique_labels)
    mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    relabeled_labels = np.array([mapping[label] for label in labels])
    return relabeled_labels


def kmeans_with_modal(features, modal_labels, k, max_iter=5):
    centroids = []
    for _ in range(k):
        modal_centroids = []
        for i in range(3):
            features_subset = features[modal_labels == i]
            random_index = torch.randint(0, features_subset.size(0), (1,))
            modal_centroids.append(features_subset[random_index])
        centroids.append(torch.stack(modal_centroids).mean(dim=0).squeeze())
    centroids = torch.stack(centroids)
    
    for _ in range(max_iter):
        distances = torch.cdist(features, centroids)
        _, assignments = torch.min(distances, dim=1)

        new_centroids = []
        for i in torch.unique(assignments).tolist():
            cluster_features = features[assignments == i]
            cluster_modal_labels = modal_labels[assignments == i]

            if len(torch.unique(cluster_modal_labels))==0:
                continue            
            
            modal_means = []
            for j in torch.unique(cluster_modal_labels).tolist():
                modal_features = cluster_features[cluster_modal_labels == j]
                modal_mean = modal_features.mean(dim=0)
                modal_means.append(modal_mean)
            new_centroid = torch.stack(modal_means).mean(dim=0)
            new_centroids.append(new_centroid)            
        new_centroids = torch.stack(new_centroids)
        
        centroids = new_centroids
    
    return relabel_labels(assignments.cpu().numpy())