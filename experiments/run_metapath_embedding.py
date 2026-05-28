import os
import sys
import yaml
import json
import time

import numpy as np

from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from loaders.dblp_loader import (
    load_dblp_dataset
)

from utils.logger import (
    get_logger
)

from core.metapath_engine import (
    MetaPathEngine
)

from core.multiplex import (
    META_PATH_MAPPING
)

from core.metrics import (
    convert_to_networkx,
    evaluate_all,
    load_author_labels,
    load_paper_labels,
    load_conf_labels,
    calculate_nmi
)

# ==========================================================
# load config
# ==========================================================

with open(
    'configs/metapath_embedding.yaml',
    'r'
) as f:

    config = yaml.safe_load(f)

EXP_NAME = config['exp_name']

# ==========================================================
# logger
# ==========================================================

logger = get_logger(
    EXP_NAME,
    f'logs/{EXP_NAME}.log'
)

# ==========================================================
# dataset
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    'datasets',
    config['dataset']
)

hetero_graph = load_dblp_dataset(
    DATASET_DIR
)

# ==========================================================
# metapath matrix
# ==========================================================

engine = MetaPathEngine(
    hetero_graph,
    logger
)

meta_path = config[
    'meta_path'
]

relations = META_PATH_MAPPING[
    meta_path
]

matrix = engine.compute_meta_path(
    relations
)

logger.info(
    f'Meta-path matrix shape='
    f'{matrix.shape}'
)

# ==========================================================
# spectral embedding
# ==========================================================

start_time = time.time()

embedding_model = SpectralEmbedding(
    n_components=config[
        'embedding'
    ]['dim'],

    affinity='precomputed'
)

embeddings = embedding_model.fit_transform(
    matrix
)

logger.info(
    f'Embedding shape='
    f'{embeddings.shape}'
)

# ==========================================================
# KMeans
# ==========================================================

kmeans = KMeans(
    n_clusters=config[
        'kmeans'
    ]['n_clusters'],

    random_state=42
)

labels = kmeans.fit_predict(
    embeddings
)

runtime = time.time() - start_time

# ==========================================================
# build communities
# ==========================================================

communities = {}

central_type = config[
    'center_type'
]

nodes = hetero_graph[
    central_type
]

for idx, label in enumerate(labels):

    communities.setdefault(
        label,
        set()
    ).add(nodes[idx])

communities = list(
    communities.values()
)

# ==========================================================
# evaluation graph
# ==========================================================

import networkx as nx

# ==========================================================
# meta-path graph
# ==========================================================

edges = engine.matrix_to_edges(
    matrix,
    src_type=config[
        'center_type'
    ],
    threshold=0
)

print(
    f'edge count = {len(edges)}'
)

G = nx.Graph()

G.add_nodes_from(nodes)

for u, v, val in edges:

    G.add_edge(
        u,
        v,
        weight=float(val)
    )

print(
    f'Graph nodes={G.number_of_nodes()} '
    f'edges={G.number_of_edges()}'
)
# ==========================================================
# evaluation
# ==========================================================

eval_results = evaluate_all(
    G,
    hetero_graph,
    communities,
    central_type,
    logger
)

# ==========================================================
# NMI
# ==========================================================

author_labels = load_author_labels(
    os.path.join(
        DATASET_DIR,
        'author_label.txt'
    )
)

author_nmi = calculate_nmi(
    communities,
    author_labels,
    node_prefix='a',
    logger=logger
)

paper_labels = load_paper_labels(
    os.path.join(
        DATASET_DIR,
        'paper_label.txt'
    )
)

paper_nmi = calculate_nmi(
    communities,
    paper_labels,
    node_prefix='p',
    logger=logger
)

conference_labels = load_conf_labels(
    os.path.join(
        DATASET_DIR,
        'conf_label.txt'
    )
)

conference_nmi = calculate_nmi(
    communities,
    conference_labels,
    node_prefix='c',
    logger=logger
)

overall_nmi = (
    author_nmi
    +
    paper_nmi
    +
    conference_nmi
) / 3

# ==========================================================
# final results
# ==========================================================

final_results = {

    'method': 'MetaPathEmbedding',

    'meta_path': meta_path,

    'modularity':
        eval_results['modularity'],

    'paper_nmi':
        paper_nmi,

    'author_nmi':
        author_nmi,

    'conference_nmi':
        conference_nmi,

    'overall_nmi':
        overall_nmi,

    'runtime':
        runtime,

    'statistics':
        eval_results['statistics']
}

# ==========================================================
# save results
# ==========================================================

RESULT_DIR = os.path.join(
    'results',
    EXP_NAME
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        RESULT_DIR,
        'results.json'
    ),
    'w'
) as f:

    json.dump(
        final_results,
        f,
        indent=4,
        default=str
    )

community_save = []

for cid, comm in enumerate(communities):

    community_save.append({

        'community_id': cid,

        'nodes': list(comm)
    })

with open(
    os.path.join(
        RESULT_DIR,
        'communities.json'
    ),
    'w'
) as f:

    json.dump(
        community_save,
        f,
        indent=4
    )

with open(
    os.path.join(
        RESULT_DIR,
        'config.json'
    ),
    'w'
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

print(final_results)