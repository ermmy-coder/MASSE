import random


# ==========================================================
# build adjacency
# ==========================================================

def build_adj(
    hetero_graph
):

    adj = {}

    for u, v, _ in hetero_graph['edges']:

        adj.setdefault(
            u,
            []
        ).append(v)

    return adj


# ==========================================================
# node type
# ==========================================================

def get_node_type(node):

    if node.startswith('a'):
        return 'author'

    if node.startswith('p'):
        return 'paper'

    if node.startswith('c'):
        return 'conference'

    if node.startswith('t'):
        return 'term'

    return None


# ==========================================================
# meta-path walk
# ==========================================================

def meta_path_walk(
    start_node,
    meta_path,
    adj,
    walk_length
):

    walk = [start_node]

    current = start_node

    for step in range(
        walk_length - 1
    ):

        expected_type = meta_path[
            step % len(meta_path)
        ]

        neighbors = adj.get(
            current,
            []
        )

        candidates = []

        for nbr in neighbors:

            if (
                get_node_type(nbr)
                ==
                expected_type
            ):

                candidates.append(nbr)

        if len(candidates) == 0:

            break

        current = random.choice(
            candidates
        )

        walk.append(current)

    return walk


# ==========================================================
# generate walks
# ==========================================================

def generate_meta_path_walks(
    hetero_graph,
    start_nodes,
    meta_path,
    walk_length=100,
    num_walks=10
):

    adj = build_adj(
        hetero_graph
    )

    walks = []

    for node in start_nodes:

        for _ in range(num_walks):

            walk = meta_path_walk(
                node,
                meta_path,
                adj,
                walk_length
            )

            walks.append(walk)

    return walks