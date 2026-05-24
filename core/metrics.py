import numpy as np
import networkx as nx
from scipy.special import comb
from collections import defaultdict


def convert_to_networkx(hetero_graph):

    G = nx.Graph()

    for node_type, nodes in hetero_graph.items():

        if node_type == 'edges':
            continue

        for node in nodes:

            G.add_node(
                node,
                node_type=node_type
            )

    for u, v, etype in hetero_graph['edges']:

        G.add_edge(
            u,
            v,
            edge_type=etype
        )

    return G


# ==========================================================
# 重叠模块度
# ==========================================================

def overlapping_modularity(
    G,
    communities
):

    m = G.number_of_edges()

    if m == 0:
        return 0

    nodes = list(G.nodes())

    A = nx.to_numpy_array(G)

    node_index = {
        n: i
        for i, n in enumerate(nodes)
    }

    O = {n: 0 for n in nodes}

    for comm in communities:

        for n in comm:

            if n in O:
                O[n] += 1

    Q = 0

    for comm in communities:

        for v in comm:

            for w in comm:

                if v == w:
                    continue

                if v not in node_index:
                    continue

                if w not in node_index:
                    continue

                i = node_index[v]
                j = node_index[w]

                kv = G.degree(v)
                kw = G.degree(w)

                Q += (
                    A[i][j]
                    - (kv * kw) / (2 * m)
                ) / (
                    max(1, O[v]) *
                    max(1, O[w])
                )

    return Q / (2 * m)


# ==========================================================
# Surprise
# ==========================================================

def calc_surprise(
    G,
    communities
):

    k = G.number_of_nodes()

    F = k * (k - 1) // 2

    n = G.number_of_edges()

    M = 0

    for comm in communities:

        s = len(comm)

        if s > 1:

            M += s * (s - 1) // 2

    p = 0

    for comm in communities:

        subg = G.subgraph(comm)

        p += subg.number_of_edges()

    min_Mn = min(M, n)

    S_sum = 0

    for j in range(p, min_Mn + 1):

        term = (
            comb(M, j)
            * comb(F - M, n - j)
            / comb(F, n)
        )

        S_sum += term

    S = -np.log10(S_sum) if S_sum > 0 else 0

    return S


# ==========================================================
# 社区内部关键词相关性
# ==========================================================

def calc_keyword_coherence(
    hetero_graph,
    communities
):

    paper2terms = defaultdict(set)

    for u, v, etype in hetero_graph['edges']:

        if etype == 'has_term':

            paper2terms[u].add(v)

    scores = []

    for comm in communities:

        papers = []

        for node in comm:

            if node.startswith('p'):

                papers.append(node)

        if len(papers) < 2:
            continue

        related = 0

        total = 0

        for i in range(len(papers)):

            for j in range(i + 1, len(papers)):

                total += 1

                if (
                    paper2terms[papers[i]]
                    &
                    paper2terms[papers[j]]
                ):

                    related += 1

        if total > 0:

            scores.append(
                related / total
            )

    return np.mean(scores) if scores else 0


# ==========================================================
# 社区规模统计
# ==========================================================

def calc_community_statistics(
    communities
):

    sizes = [
        len(c)
        for c in communities
    ]

    return {
        'num_communities': len(communities),
        'avg_size': np.mean(sizes),
        'max_size': np.max(sizes),
        'min_size': np.min(sizes)
    }


# ==========================================================
# 总评估函数
# ==========================================================

def evaluate_all(
    hetero_graph,
    communities,
    logger=None
):

    G = convert_to_networkx(
        hetero_graph
    )

    modularity = overlapping_modularity(
        G,
        communities
    )

    surprise = calc_surprise(
        G,
        communities
    )

    coherence = calc_keyword_coherence(
        hetero_graph,
        communities
    )

    stats = calc_community_statistics(
        communities
    )

    results = {
        'modularity': modularity,
        'surprise': surprise,
        'keyword_coherence': coherence,
        'statistics': stats
    }

    if logger:

        logger.info(
            f'[EVAL] '
            f'Modularity={modularity:.4f}'
        )

        logger.info(
            f'[EVAL] '
            f'Surprise={surprise:.4f}'
        )

        logger.info(
            f'[EVAL] '
            f'Keyword coherence={coherence:.4f}'
        )

        logger.info(
            f'[EVAL] '
            f'Community stats={stats}'
        )

    return results


from sklearn.metrics import normalized_mutual_info_score


# def load_paper_labels(label_path):

#     labels = {}

#     with open(label_path, 'r', encoding='utf-8') as f:

#         for idx, line in enumerate(f):

#             line = line.strip()

#             if line == '':
#                 continue

#             labels[f'p{idx}'] = line

#     return labels

def load_paper_labels(label_path):

    labels = {}

    with open(
        label_path,
        'r',
        encoding='utf-8'
    ) as f:

        for line in f:

            line = line.strip()

            if line == '':
                continue

            parts = line.split()

            # ==========================================
            # paper_id label
            # ==========================================

            if len(parts) < 2:
                continue

            paper_id = parts[0]

            label = parts[1]

            labels[f'p{paper_id}'] = label

    return labels

def load_author_labels(label_path):

    labels = {}

    with open(
        label_path,
        'r',
        encoding='utf-8'
    ) as f:

        for line in f:

            parts = line.strip().split('\t')

            if len(parts) < 2:
                continue

            author_id = parts[0]

            label = parts[1]

            labels[f'a{author_id}'] = label

    return labels


def calculate_nmi(
    communities,
    ground_truth_labels,
    node_prefix,
    logger=None
):

    pred_nodes = []
    pred_labels = []

    gt_labels = []

    for cid, comm in enumerate(communities):

        for node in comm:

            if not node.startswith(node_prefix):
                continue

            if node not in ground_truth_labels:
                continue

            pred_nodes.append(node)

            pred_labels.append(cid)

            gt_labels.append(
                ground_truth_labels[node]
            )

    if logger:
    
        logger.info(
            f'[NMI DEBUG] '
            f'matched_nodes='
            f'{len(pred_labels)}'
        )
    
    if len(pred_labels) == 0:
    
        return 0

    nmi = normalized_mutual_info_score(
        gt_labels,
        pred_labels
    )

    if logger:

        logger.info(
            f'[NMI] {nmi:.4f}'
        )

    return nmi