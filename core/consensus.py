import networkx as nx
from collections import defaultdict
from cdlib import algorithms
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


# ==========================================================
# convert communities -> labels
# ==========================================================

def community_to_label_map(comms):

    node2label = {}

    for cid, comm in enumerate(comms):

        for node in comm:

            node2label[node] = cid

    return node2label
# ==========================================================
# MASSE:
# meta-path interaction graph
# ==========================================================

def compute_meta_path_interaction(
    layer_comms,
    logger
):

    layers = list(
        layer_comms.keys()
    )

    interaction = {}

    for li in layers:

        interaction[li] = {}

        for lj in layers:

            if li == lj:

                interaction[li][lj] = 1.0

                continue

            comms_i = layer_comms[li]
            comms_j = layer_comms[lj]

            map_i = community_to_label_map(
                comms_i
            )

            map_j = community_to_label_map(
                comms_j
            )

            common_nodes = list(
                set(map_i.keys())
                &
                set(map_j.keys())
            )

            if len(common_nodes) == 0:

                score = 0.0

            else:

                labels_i = [
                    map_i[n]
                    for n in common_nodes
                ]

                labels_j = [
                    map_j[n]
                    for n in common_nodes
                ]

                score = (
                    normalized_mutual_info_score(
                        labels_i,
                        labels_j
                    )
                )

            interaction[li][lj] = score

            logger.info(
                f'[INTERACTION] '
                f'{li} <-> {lj} '
                f'= {score:.4f}'
            )

    return interaction
#add

from sklearn.cluster import SpectralClustering


def build_semantic_subspaces(
    interaction,
    logger,
    n_clusters=2,
    random_state=42
):

    layers = list(
        interaction.keys()
    )

    n = len(layers)

    M = np.zeros(
        (n, n)
    )

    for i, l1 in enumerate(layers):

        for j, l2 in enumerate(layers):

            M[i, j] = interaction[l1][l2]

    labels = SpectralClustering(

        n_clusters=min(
            n_clusters,
            n
        ),

        affinity='precomputed',

        random_state=random_state

    ).fit_predict(M)

    subspaces = {}

    for layer, label in zip(
        layers,
        labels
    ):

        subspaces.setdefault(
            label,
            []
        ).append(layer)

    logger.info(
        f'[SUBSPACE] {subspaces}'
    )

    return subspaces
###



def detect_seed_communities(
    multiplex,
    logger,
    config=None
):

    # ======================================================
    # Step1:
    # detect communities in each layer
    # ======================================================

    layer_comms = {}

    # ======================================================
    # node -> community_id
    # ======================================================

    layer_node2comm = {}

    for layer, obj in multiplex.items():

        G = obj['graph']
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        
            logger.warning(
                f'[SKIP] {layer} is empty.'
            )
        
            continue

        logger.info(
            f'Running community detection on {layer}'
        )

        comms = algorithms.louvain(
            G
        ).communities

        logger.info(
            f'{layer} communities='
            f'{len(comms)}'
        )

        layer_comms[layer] = comms

        # ==================================================
        # build node->community mapping
        # ==================================================

        node2comm = {}

        for cid, comm in enumerate(comms):

            for node in comm:

                node2comm[node] = cid

        layer_node2comm[layer] = node2comm

    # ======================================================
    # Step2:
    # build consensus graph
    # ======================================================

    consensus = nx.Graph()

    # ======================================================
    # collect all candidate edges
    # ======================================================

    candidate_edges = set()

    for layer, obj in multiplex.items():

        G = obj['graph']

        for u, v in G.edges():

            if u > v:
                u, v = v, u

            candidate_edges.add(
                (u, v)
            )

    logger.info(
        f'Candidate edges='
        f'{len(candidate_edges)}'
    )

    # ======================================================
    # add nodes
    # ======================================================

    nodes = set()

    for layer, obj in multiplex.items():

        nodes.update(
            obj['graph'].nodes()
        )

    consensus.add_nodes_from(nodes)

    layers = list(layer_node2comm.keys())

    # ======================================================
    # Step3:
    # edge-wise consensus aggregation
    # ======================================================

    processed = 0

    for u, v in candidate_edges:

        total_weight = 0

        agree_count = 0

        for layer in layers:

            G = multiplex[layer]['graph']

            node2comm = layer_node2comm[layer]

            # ==============================================
            # edge existence
            # ==============================================

            if not G.has_edge(u, v):
                continue

            # ==============================================
            # same community?
            # ==============================================

            same = (
                node2comm.get(u, -1)
                ==
                node2comm.get(v, -1)
            )

            if not same:
                continue

            # ==============================================
            # weighted contribution
            # ==============================================

            edge_weight = G[u][v].get(
                'weight',
                0
            )

            layer_weight = multiplex[layer][
                'weight'
            ]

            total_weight += (
                edge_weight
                *
                layer_weight
            )

            agree_count += 1

        # ==================================================
        # consensus score
        # ==================================================

        min_agree = config[
            'consensus'
        ]['min_agree']
        
        if agree_count >= min_agree:
            normalize_mode = config[
                'consensus'
            ]['normalize']
            
            if normalize_mode == 'all_layers':
            
                consensus_score = (
                    total_weight
                    / len(layers)
                )
            
            else:
            
                consensus_score = (
                    total_weight
                    / agree_count
                )
            consensus.add_edge(
                u,
                v,
                weight=consensus_score
            )

        processed += 1

        if processed % 100000 == 0:

            logger.info(
                f'Processed '
                f'{processed}/'
                f'{len(candidate_edges)} '
                f'candidate edges'
            )

    # ======================================================
    # final consensus graph stats
    # ======================================================

    logger.info(
        f'Consensus graph '
        f'nodes='
        f'{consensus.number_of_nodes()} '
        f'edges='
        f'{consensus.number_of_edges()}'
    )

    # ======================================================
    # final seed communities
    # ======================================================

    logger.info(
        'Running Louvain on consensus graph'
    )

    seed_comms = algorithms.louvain(
        consensus,
        weight='weight'#add
    ).communities

    logger.info(
        f'Seed communities='
        f'{len(seed_comms)}'
    )

    return ([
        set(c)
        for c in seed_comms
        ],
        consensus
    )
# ==========================================================
# MASSE seed communities
# ==========================================================

def detect_masse_seed_communities(
    multiplex,
    logger,
    config=None
):

    # ======================================================
    # Step1:
    # detect communities in each layer
    # ======================================================

    layer_comms = {}

    layer_node2comm = {}

    for layer, obj in multiplex.items():

        G = obj['graph']

        if (
            G.number_of_nodes() == 0
            or
            G.number_of_edges() == 0
        ):

            continue

        comms = algorithms.louvain(
            G
        ).communities

        layer_comms[layer] = comms

        node2comm = {}

        for cid, comm in enumerate(comms):

            for node in comm:

                node2comm[node] = cid

        layer_node2comm[layer] = node2comm

    # ======================================================
    # Step2:
    # interaction graph
    # ======================================================

    if config['ablation']['disable_interaction']:

        interaction = {}

        layers = list(layer_comms.keys())

        for li in layers:

            interaction[li] = {}

            for lj in layers:

                interaction[li][lj] = (
                    1.0 if li == lj else 0.0
                )

    else:

        interaction = (
            compute_meta_path_interaction(
                layer_comms,
                logger
            )
        )
    #add
    if config['ablation']['disable_subspace']:

        subspaces = {
            0:list(
                layer_comms.keys()
            )
        }

    else:

        subspaces = build_semantic_subspaces(
            interaction,
            logger,
            n_clusters=config[
                'masse'
            ]['subspace'][
                'n_clusters'
            ],
            random_state=config[
                'seed'
            ]
        )
    # ======================================================
    # Step3:
    # Local Consensus Graph
    # ======================================================

    local_communities = []

    for sid, layers_in_space in subspaces.items():

        logger.info(
            f'[LOCAL] '
            f'Subspace {sid}: '
            f'{layers_in_space}'
        )

        local_graph = nx.Graph()

        for layer in layers_in_space:

            G = multiplex[layer]['graph']

            reliability = multiplex[
                layer
            ].get(
                'reliability',
                1.0
            )

            for u, v, data in G.edges(
                data=True
            ):

                w = data.get(
                    'weight',
                    1.0
                )

                w *= reliability

                if local_graph.has_edge(
                    u,
                    v
                ):

                    local_graph[u][v][
                        'weight'
                    ] += w

                else:

                    local_graph.add_edge(
                        u,
                        v,
                        weight=w
                    )

        comms = algorithms.louvain(
            local_graph,
            weight='weight'
        ).communities

        logger.info(
            f'[LOCAL] communities='
            f'{len(comms)}'
        )

        local_communities.append(
            [
                set(c)
                for c in comms
            ]
        )

    # ======================================================
    # Step4:
    # Evidence Fusion
    # ======================================================
    if config['ablation']['disable_fusion']:

        final_comms = local_communities[0]

        consensus = local_graph

        return (
            [
                set(c)
                for c in final_comms
            ],
            consensus
        )

    phi = defaultdict(float)

    for comms in local_communities:

        for comm in comms:

            nodes = list(comm)

            for i in range(
                len(nodes)
            ):

                for j in range(
                    i + 1,
                    len(nodes)
                ):

                    u = nodes[i]
                    v = nodes[j]

                    if u > v:
                        u, v = v, u

                    phi[(u, v)] += 1

    num_spaces = max(
        1,
        len(local_communities)
    )

    for k in phi:

        phi[k] /= num_spaces

    # ======================================================
    # Step5:
    # Final Consensus Graph
    # ======================================================

    consensus = nx.Graph()

    for (u, v), score in phi.items():

        if score <= 0:
            continue

        consensus.add_edge(
            u,
            v,
            weight=score
        )

    logger.info(
        f'MASSE consensus '
        f'nodes={consensus.number_of_nodes()} '
        f'edges={consensus.number_of_edges()}'
    )

    # ======================================================
    # Step6:
    # Final Community Detection
    # ======================================================

    final_comms = algorithms.louvain(
        consensus,
        weight='weight'
    ).communities

    logger.info(
        f'MASSE communities='
        f'{len(final_comms)}'
    )

#     # ======================================================
#     # Step3:
#     # consensus graph
#     # ======================================================

#     consensus = nx.Graph()

#     candidate_edges = set()

#     for layer, obj in multiplex.items():

#         G = obj['graph']

#         for u, v in G.edges():

#             if u > v:
#                 u, v = v, u

#             candidate_edges.add(
#                 (u, v)
#             )

#     nodes = set()

#     for layer, obj in multiplex.items():

#         nodes.update(
#             obj['graph'].nodes()
#         )

#     consensus.add_nodes_from(nodes)

#     layers = list(
#         layer_node2comm.keys()
#     )

#     for u, v in candidate_edges:

#         total_weight = 0

#         agree_count = 0

#         for layer in layers:

#             G = multiplex[layer]['graph']

#             node2comm = (
#                 layer_node2comm[layer]
#             )

#             if not G.has_edge(u, v):

#                 continue

#             same = (
#                 node2comm.get(u, -1)
#                 ==
#                 node2comm.get(v, -1)
#             )

#             if not same:

#                 continue

#             edge_weight = G[u][v].get(
#                 'weight',
#                 0
#             )

#             reliability = multiplex[
#                 layer
#             ].get(
#                 'reliability',
#                 1.0
#             )
#             #delete

# #             aggregation_mode = config[
# #                 'masse'
# #             ]['interaction'][
# #                 'aggregation'
# #             ]
            
# #             interaction_values = list(
# #                 interaction[layer].values()
# #             )
            
# #             if aggregation_mode == 'mean':
            
# #                 interaction_score = np.mean(
# #                     interaction_values
# #                 )
            
# #             elif aggregation_mode == 'max':
            
# #                 interaction_score = np.max(
# #                     interaction_values
# #                 )
            
# #             elif aggregation_mode == 'topk':
            
# #                 topk = config[
# #                     'masse'
# #                 ]['interaction'][
# #                     'topk'
# #                 ]
            
# #                 sorted_scores = sorted(
# #                     interaction_values,
# #                     reverse=True
# #                 )
            
# #                 selected = sorted_scores[:topk]
            
# #                 interaction_score = np.mean(
# #                     selected
# #                 )
            
# #             else:
            
# #                 interaction_score = np.mean(
# #                     interaction_values
# #                 )

# #             alpha = config[
# #                 'masse'
# #             ]['consensus'][
# #                 'semantic_alpha'
# #             ]
            
# #             semantic_weight = (
# #                 reliability
# #                 *
# #                 (
# #                     interaction_score
# #                     ** alpha
# #                 )
# #             )

#             total_weight += (
#                 edge_weight
#                 *
#                 semantic_weight
#             )

#             agree_count += 1

#         min_agree = config[
#             'masse'
#         ]['consensus'][
#             'min_agree'
#         ]
        
#         if agree_count >= min_agree:

#             normalize_mode = config[
#                 'masse'
#             ]['consensus'][
#                 'normalize'
#             ]
            
#             if normalize_mode == 'all_layers':
            
#                 consensus_score = (
#                     total_weight
#                     / len(layers)
#                 )
            
#             else:
            
#                 consensus_score = (
#                     total_weight
#                     / agree_count
#                 )

#             consensus.add_edge(
#                 u,
#                 v,
#                 weight=consensus_score
#             )

#     logger.info(
#         f'MASSE consensus '
#         f'nodes={consensus.number_of_nodes()} '
#         f'edges={consensus.number_of_edges()}'
#     )

#     final_comms = algorithms.louvain(
#         consensus
#     ).communities

#     logger.info(
#         f'MASSE communities='
#         f'{len(final_comms)}'
#     )

    return (
        [
            set(c)
            for c in final_comms
        ],
        consensus
    )