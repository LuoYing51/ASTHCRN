import numpy as np
import torch

def build_hypergraph_distance(distance_matrix, KNN_num, distance_threshold, device):
    num_sites = distance_matrix.shape[0]
    sorted_indices = np.argsort(distance_matrix, axis=1)

    candidate_mask = (distance_matrix <= distance_threshold) & (distance_matrix > 0)

    row_indices = np.arange(num_sites)[:, None]
    knn_mask = np.zeros_like(distance_matrix, dtype=bool)
    knn_mask[row_indices, sorted_indices[:, 1:KNN_num + 1]] = True

    final_mask = candidate_mask & knn_mask

    H_list_un_sorted = []
    seen = set()
    for i in range(num_sites):
        neighbors = np.where(final_mask[i])[0].tolist()
        hyperedge = [i] + neighbors
        hyperedge_tuple = tuple(hyperedge)
        if hyperedge_tuple not in seen:
            seen.add(hyperedge_tuple)
            H_list_un_sorted.append(hyperedge)

    if H_list_un_sorted:
        H_list = list({tuple(sorted(e)) for e in H_list_un_sorted})  # 排序 升序（默认）
    else:
        H_list = []

    num_edges = len(H_list)

    H = np.zeros((num_sites, num_edges))
    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            H[node, edge_idx] = 1

    node_indices = []
    hyperedge_indices = []

    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            node_indices.append(node)
            hyperedge_indices.append(edge_idx)
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
    hyperedge_index = hyperedge_index.to(device)

    return H_list, H, hyperedge_index


def build_hypergraph_distance_get_centers(distance_matrix, KNN_num, distance_threshold, device):
    num_sites = distance_matrix.shape[0]
    sorted_indices = np.argsort(distance_matrix, axis=1)

    candidate_mask = (distance_matrix <= distance_threshold) & (distance_matrix > 0)

    row_indices = np.arange(num_sites)[:, None]
    knn_mask = np.zeros_like(distance_matrix, dtype=bool)
    knn_mask[row_indices, sorted_indices[:, 1:KNN_num + 1]] = True

    final_mask = candidate_mask & knn_mask

    H_list_un_sorted = []

    for i in range(num_sites):
        neighbors = np.where(final_mask[i])[0].tolist()
        hyperedge = [i] + neighbors
        H_list_un_sorted.append(hyperedge)

    seen_hyperedge = set()
    H_list = []
    centers_list = []
    redundant_hyperedges = []
    redundant_centers = []

    for hyperedge in H_list_un_sorted:
        sorted_hyperedge = tuple(sorted(hyperedge))
        original_center = hyperedge[0]

        if sorted_hyperedge not in seen_hyperedge:
            seen_hyperedge.add(sorted_hyperedge)
            H_list.append(hyperedge)
            centers_list.append(original_center)
        else:
            redundant_hyperedges.append(hyperedge)
            redundant_centers.append(original_center)

    num_edges = len(H_list)

    H = np.zeros((num_sites, num_edges))
    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            H[node, edge_idx] = 1

    node_indices = []
    hyperedge_indices = []

    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            node_indices.append(node)
            hyperedge_indices.append(edge_idx)
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)
    hyperedge_index = hyperedge_index.to(device)

    return H_list, H, hyperedge_index, centers_list, redundant_hyperedges, redundant_centers


def build_hypergraph_distance_similarity(
        s_KNN_num,
        similarity_matrix,
        similarity_threshold,
        distance_matrix,
        distance_threshold,
        device='cuda'):
    H_list_ = []
    H_list = []
    num_sites = similarity_matrix.shape[0]
    for i in range(num_sites):
        similarities = similarity_matrix[i]
        distances = distance_matrix[i]

        eligible_indices = np.where(
            (similarities.cpu() >= similarity_threshold) &
            (distances <= distance_threshold) &
            (np.arange(num_sites) != i)
        )[0]
        if len(eligible_indices) >= 1:
            sorted_scores, sorted_indices = torch.sort(-similarities.detach()[eligible_indices], descending=False,
                                                       stable=True)
            sorted_indices = eligible_indices[sorted_indices.cpu()]
            neighbor_indices = sorted_indices[:s_KNN_num]
        else:
            neighbor_indices = np.array([])

        if len(neighbor_indices) > 0:
            hyperedge = [i] + neighbor_indices.tolist()
        else:
            hyperedge = [i]
        H_list_.append(hyperedge)

    seen = set()
    for sublist in H_list_:
        sorted_sublist = tuple(sorted(set(sublist)))
        if sorted_sublist not in seen:
            seen.add(sorted_sublist)
            H_list.append(list(sorted_sublist))

    num_edges = len(H_list)
    H = np.zeros((num_sites, num_edges))
    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            H[node, edge_idx] = 1

    H = torch.tensor(H, dtype=torch.float).to(device)

    node_indices = []
    hyperedge_indices = []
    for edge_idx, hyperedge in enumerate(H_list):
        for node in hyperedge:
            node_indices.append(node)
            hyperedge_indices.append(edge_idx)
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long).to(device)

    return H, hyperedge_index