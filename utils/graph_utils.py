from collections import defaultdict
import networkx as nx


# ==========================================================
# 构建邻接表
# ==========================================================

def build_adjacency_list(
    hetero_graph
):

    adj = defaultdict(set)

    for u, v, _ in hetero_graph['edges']:

        adj[u].add(v)

        adj[v].add(u)

    return adj


# ==========================================================
# 节点类型统计
# ==========================================================

def print_graph_statistics(
    hetero_graph,
    logger=None
):

    print('\n========== GRAPH STATS ==========')

    for node_type, nodes in hetero_graph.items():

        if node_type == 'edges':
            continue

        msg = (
            f'{node_type}: '
            f'{len(nodes)}'
        )

        print(msg)

        if logger:
            logger.info(msg)

    edge_types = defaultdict(int)

    for _, _, etype in hetero_graph['edges']:

        edge_types[etype] += 1

    print('\n========== EDGE TYPES ==========')

    for etype, count in edge_types.items():

        msg = f'{etype}: {count}'

        print(msg)

        if logger:
            logger.info(msg)

    print(
        f'\nTotal edges: '
        f'{len(hetero_graph["edges"])}'
    )


# ==========================================================
# 转换为networkx
# ==========================================================

def hetero_to_networkx(
    hetero_graph
):

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
# 检查孤立节点
# ==========================================================

def check_isolated_nodes(
    hetero_graph,
    logger=None
):

    G = hetero_to_networkx(
        hetero_graph
    )

    isolated = list(
        nx.isolates(G)
    )

    msg = (
        f'Isolated nodes: '
        f'{len(isolated)}'
    )

    print(msg)

    if logger:
        logger.warning(msg)

    return isolated


# ==========================================================
# 删除孤立节点
# ==========================================================

def remove_isolated_nodes(
    hetero_graph
):

    G = hetero_to_networkx(
        hetero_graph
    )

    isolated = set(
        nx.isolates(G)
    )

    new_graph = {}

    for node_type, nodes in hetero_graph.items():

        if node_type == 'edges':
            continue

        new_graph[node_type] = [
            n for n in nodes
            if n not in isolated
        ]

    new_edges = []

    for u, v, etype in hetero_graph['edges']:

        if u in isolated:
            continue

        if v in isolated:
            continue

        new_edges.append(
            (u, v, etype)
        )

    new_graph['edges'] = new_edges

    return new_graph