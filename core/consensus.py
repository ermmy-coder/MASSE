import networkx as nx
from collections import defaultdict
from cdlib import algorithms


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

    return [
        set(c)
        for c in seed_comms
    ]