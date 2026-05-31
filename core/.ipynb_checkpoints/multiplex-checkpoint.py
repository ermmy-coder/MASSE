import networkx as nx
import numpy as np

from core.metapath_engine import (
    MetaPathEngine
)

# ==========================================================
# meta-path relation mapping
# ==========================================================

META_PATH_MAPPING = {

    # ======================================================
    # author-centric
    # ======================================================

    'A-P-A': [
        'be_written',
        'writes'
    ],

    'A-P-T-P-A': [
        'be_written',
        'has_term',
        'term_in',
        'writes'
    ],

    'A-P-C-P-A': [
        'be_written',
        'published_in',
        'publish',
        'writes'
    ],

    # ======================================================
    # paper-centric
    # ======================================================

    'P-A-P': [
        'writes',
        'be_written'
    ],

    'P-T-P': [
        'has_term',
        'term_in'
    ],

    'P-C-P': [
        'published_in',
        'publish'
    ]
}

# ==========================================================
# MASSE:
# meta-path reliability
# ==========================================================

def compute_meta_path_reliability(
    G,
    config
):

    weights = []

    for _, _, data in G.edges(data=True):

        w = data.get(
            'weight',
            1.0
        )

        weights.append(w)

    if len(weights) == 0:

        return 0.0

    weights = np.array(weights)

    p = weights / (
        weights.sum() + 1e-12
    )

    entropy = -np.sum(
        p * np.log(
            p + 1e-12
        )
    )

    tau = config[
        'masse'
    ]['reliability'][
        'entropy_temperature'
    ]
    
    # reliability = np.exp(
    #     -entropy / tau
    # )
    reliability = 1 / (
        1 + entropy
    )
    min_r = config[
        'masse'
    ]['reliability'][
        'min_reliability'
    ]
    
    reliability = max(
        reliability,
        min_r
    )

    return float(reliability)


# ==========================================================
# build multiplex network
# ==========================================================

def build_multiplex_network(
    hetero_graph,
    central_type,
    meta_paths,
    logger,
    config=None
):

    multiplex = {}

    engine = MetaPathEngine(
        hetero_graph,
        logger
    )

    for meta_path, path_weight in meta_paths.items():

        logger.info(
            f'\n========== '
            f'Building {meta_path} '
            f'=========='
        )

        if meta_path not in META_PATH_MAPPING:

            logger.warning(
                f'{meta_path} '
                f'not implemented'
            )

            continue

        relations = META_PATH_MAPPING[
            meta_path
        ]

        # ==================================================
        # sparse matrix multiplication
        # ==================================================

        matrix = engine.compute_meta_path(
            relations
        )

        logger.info(
            f'{meta_path} '
            f'matrix shape={matrix.shape}'
        )

        # ==================================================
        # matrix -> graph
        # ==================================================
        
        # ==========================================================
        # config thresholds
        # ==========================================================
        
        thresholds = config.get(
            'thresholds',
            {}
        )
        
        edges = engine.matrix_to_edges(
            matrix,
            central_type,
            threshold=thresholds.get(
                meta_path,
                1
            )
        )

        G = nx.Graph()

        total_weight = 0

        for u, v, val in edges:

            # ==============================================
            # beta path weighting
            # ==============================================

            middle_nodes = max(
                1,
                len(relations) - 1
            )

            beta = 1 / middle_nodes

            final_weight = val * beta

            G.add_edge(
                u,
                v,
                weight=final_weight
            )

            total_weight += final_weight

        logger.info(
            f'{meta_path} '
            f'nodes={G.number_of_nodes()} '
            f'edges={G.number_of_edges()} '
            f'total_weight={total_weight:.4f}'
        )
        if G.number_of_edges() == 0:
        
            logger.warning(
                f'[SKIP] {meta_path} generated empty graph.'
            )

        multiplex[meta_path] = {
            'graph': G,
            'weight': path_weight
        }

    return multiplex
# ==========================================================
# MASSE multiplex
# ==========================================================

def build_masse_multiplex(
    hetero_graph,
    central_type,
    meta_paths,
    logger,
    config=None
):

    multiplex = build_multiplex_network(
        hetero_graph,
        central_type,
        meta_paths,
        logger,
        config
    )

    logger.info(
        '\n========== '
        'MASSE Reliability '
        '=========='
    )

    for meta_path in multiplex:

        G = multiplex[meta_path]['graph']

        if config['ablation']['disable_reliability']:

            reliability = 1.0

        else:

            reliability = (
                compute_meta_path_reliability(
                    G,
                    config
                )
            )

        multiplex[meta_path][
            'reliability'
        ] = reliability

        logger.info(
            f'{meta_path} '
            f'reliability='
            f'{reliability:.6f}'
        )

    return multiplex