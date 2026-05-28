import os
import sys
import yaml
import json
import time

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

from core.community_detection import (
    run_leiden
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
    'configs/leiden.yaml',
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
# graph
# ==========================================================

G = convert_to_networkx(
    hetero_graph
)

logger.info(
    f'Graph nodes={G.number_of_nodes()} '
    f'edges={G.number_of_edges()}'
)

# ==========================================================
# runtime
# ==========================================================

start_time = time.time()

# ==========================================================
# Leiden
# ==========================================================

communities = run_leiden(

    G,

    resolution=config[
        'leiden'
    ]['resolution'],

    weights=config[
        'leiden'
    ]['weight']
)

runtime = time.time() - start_time

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

    'method': 'Leiden',

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