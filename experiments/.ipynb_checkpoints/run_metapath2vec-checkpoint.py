import os
import sys
import yaml
import json
import time

import numpy as np

from gensim.models import Word2Vec

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

from core.metapath_walk import (
    generate_meta_path_walks
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
    'configs/metapath2vec.yaml',
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
# runtime
# ==========================================================

start_time = time.time()

# ==========================================================
# generate walks
# ==========================================================

walks = generate_meta_path_walks(

    hetero_graph,

    hetero_graph[
        config['center_type']
    ],

    meta_path=config[
        'meta_path'
    ],

    walk_length=config[
        'walk'
    ]['length'],

    num_walks=config[
        'walk'
    ]['num_walks']
)

logger.info(
    f'Generated walks='
    f'{len(walks)}'
)

# ==========================================================
# Word2Vec
# ==========================================================

model = Word2Vec(

    walks,

    vector_size=config[
        'word2vec'
    ]['dimensions'],

    window=config[
        'word2vec'
    ]['window_size'],

    min_count=1,

    workers=config[
        'word2vec'
    ]['workers'],

    sg=1
)

logger.info(
    'Word2Vec training finished'
)

# ==========================================================
# embeddings
# ==========================================================

embeddings = []

valid_nodes = []

for node_type, node_list in hetero_graph.items():

    if node_type == 'edges':
        continue

    for node in node_list:

        if node in model.wv:

            embeddings.append(
                model.wv[node]
            )

            valid_nodes.append(
                node
            )

embeddings = np.array(
    embeddings
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
# communities
# ==========================================================

communities = {}

for node, label in zip(
    valid_nodes,
    labels
):

    communities.setdefault(
        int(label),
        set()
    ).add(node)

communities = list(
    communities.values()
)

covered_nodes = set()

for comm in communities:

    covered_nodes.update(comm)

all_nodes = set()

for node_type, node_list in hetero_graph.items():

    if node_type == "edges":
        continue

    all_nodes.update(node_list)

missing_nodes = all_nodes - covered_nodes

logger.info(
    f'Covered nodes={len(covered_nodes)}'
)

logger.info(
    f'Missing nodes={len(missing_nodes)}'
)

logger.info(
    f'Total nodes={len(all_nodes)}'
)

for node in missing_nodes:

    communities.append(
        {node}
    )

# ==========================================================
# evaluation graph
# ==========================================================

# import networkx as nx

# G = nx.Graph()

# for comm in communities:

#     for u in comm:

#         G.add_node(u)

# for comm in communities:

#     comm = list(comm)

#     for i in range(len(comm)):

#         for j in range(i + 1, len(comm)):

#             G.add_edge(
#                 comm[i],
#                 comm[j]
#             )

G = convert_to_networkx(
    hetero_graph
)

# ==========================================================
# evaluation
# ==========================================================

eval_results = evaluate_all(
    G,
    hetero_graph,
    communities,
    config['center_type'],
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

    'method': 'MetaPath2Vec',

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

# ==========================================================
# save communities
# ==========================================================

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

# ==========================================================
# save config
# ==========================================================

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

logger.info(
    'Saved results...'
)

print(final_results)