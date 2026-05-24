import matplotlib.pyplot as plt
import networkx as nx
import os


# ==========================================================
# 可视化multiplex layer
# ==========================================================

def visualize_layer(
    G,
    title,
    save_path=None
):

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G)

    weights = [
        G[u][v]['weight']
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=300
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=weights
    )

    plt.title(title)

    if save_path:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300
        )

    plt.close()


# ==========================================================
# 可视化社区
# ==========================================================

def visualize_communities(
    G,
    communities,
    title,
    save_path=None
):

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G)

    color_map = {}

    for idx, comm in enumerate(communities):

        for node in comm:

            color_map[node] = idx

    colors = []

    for node in G.nodes():

        colors.append(
            color_map.get(node, -1)
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        cmap=plt.cm.Set3,
        node_size=300
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    plt.title(title)

    if save_path:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300
        )

    plt.close()


# ==========================================================
# 指标柱状图
# ==========================================================

def plot_metrics(
    result_dict,
    save_path=None
):

    metrics = [
        'modularity',
        'surprise',
        'keyword_coherence'
    ]

    names = list(result_dict.keys())

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = range(len(metrics))

    width = 0.2

    for idx, method in enumerate(names):

        values = [
            result_dict[method][m]
            for m in metrics
        ]

        offset = [
            i + idx * width
            for i in x
        ]

        ax.bar(
            offset,
            values,
            width=width,
            label=method
        )

    ax.set_xticks(
        [i + width for i in x]
    )

    ax.set_xticklabels(metrics)

    ax.legend()

    plt.title(
        'Community Detection Performance'
    )

    if save_path:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300
        )

    plt.close()