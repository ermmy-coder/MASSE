import time
import yaml

from core.metrics import (
    convert_to_networkx,
    evaluate_all,
    calculate_nmi,
    load_author_labels,
    load_paper_labels
)

from core.community_detection import (
    run_louvain
)

from data.dblp_loader import (
    load_dblp_dataset
)


# ==========================================================
# Louvain baseline
# ==========================================================

def run_louvain_baseline(config_path):

    # ======================================================
    # load config
    # ======================================================

    with open(
        config_path,
        'r',
        encoding='utf-8'
    ) as f:

        config = yaml.safe_load(f)

    # ======================================================
    # load dataset
    # ======================================================

    dataset_dir = '../datasets/DBLP/'

    hetero_graph = load_dblp_dataset(
        dataset_dir
    )

    # ======================================================
    # hetero -> homogeneous graph
    # ======================================================

    G = convert_to_networkx(
        hetero_graph
    )

    print(
        f'\nGraph: '
        f'nodes={G.number_of_nodes()} '
        f'edges={G.number_of_edges()}'
    )

    # ======================================================
    # run Louvain
    # ======================================================

    start_time = time.time()

    communities = run_louvain(
        G
    )

    runtime = (
        time.time()
        - start_time
    )

    print(
        f'\nLouvain runtime='
        f'{runtime:.4f}s'
    )

    # ======================================================
    # evaluation
    # ======================================================

    central_type = config[
        'central_type'
    ]

    results = evaluate_all(
        G,
        hetero_graph,
        communities,
        central_type,
        logger=None
    )

    # ======================================================
    # load labels
    # ======================================================

    if central_type == 'author':

        gt_labels = load_author_labels(
            '../datasets/DBLP/label/author_label.txt'
        )

        node_prefix = 'a'

    else:

        gt_labels = load_paper_labels(
            '../datasets/DBLP/label/paper_label.txt'
        )

        node_prefix = 'p'

    # ======================================================
    # NMI
    # ======================================================

    nmi = calculate_nmi(
        communities,
        gt_labels,
        node_prefix
    )

    results['nmi'] = nmi

    results['runtime'] = runtime

    # ======================================================
    # print results
    # ======================================================

    print('\n========== RESULTS ==========')

    print(
        f'Modularity='
        f'{results["modularity"]:.4f}'
    )

    print(
        f'Surprise='
        f'{results["surprise"]:.4f}'
    )

    print(
        f'NMI='
        f'{results["nmi"]:.4f}'
    )

    print(
        f'Runtime='
        f'{results["runtime"]:.4f}s'
    )

    print(
        f'Communities='
        f'{results["statistics"]["num_communities"]}'
    )

    return communities, results


# ==========================================================
# main
# ==========================================================

if __name__ == "__main__":

    run_louvain_baseline(
        '../configs/louvain.yaml'
    )