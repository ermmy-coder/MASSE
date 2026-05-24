def build_adj(graph):

    adj = {}

    for u, v, _ in graph['edges']:

        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    return adj


def seed_expansion(
    hetero_graph,
    seed_comms,
    central_type,
    logger
):

    adj = build_adj(hetero_graph)

    results = []

    for cid, seed in enumerate(seed_comms):

        logger.info(
            f'Expanding community {cid}'
        )

        comm = {
            central_type: set(seed)
        }

        for t in hetero_graph:

            if t == 'edges':
                continue

            if t != central_type:
                comm[t] = set()

        for t, nodes in hetero_graph.items():

            if t in ['edges', central_type]:
                continue

            score_list = []

            for node in nodes:

                neighbors = adj.get(node, set())

                sim = len(
                    neighbors &
                    set().union(*comm.values())
                )

                score_list.append(
                    (node, sim)
                )

            score_list.sort(
                key=lambda x: -x[1]
            )

            logger.debug(
                f'[EXPANSION] '
                f'{t} sorted scores='
                f'{score_list[:10]}'
            )

            for node, sim in score_list:

                if sim > 0:

                    comm[t].add(node)

        results.append(comm)

    return results