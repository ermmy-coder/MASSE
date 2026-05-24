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

from core.multiplex import (
    build_multiplex_network
)

from core.consensus import (
    detect_seed_communities
)

from core.expansion import (
    seed_expansion
)

from core.metrics import (
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
    'configs/hete_mese.yaml',
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
# multiplex
# ==========================================================

multiplex = build_multiplex_network(
    hetero_graph,
    config['center_type'],
    config['meta_paths'],
    logger,
    config
)

# ==========================================================
# consensus
# ==========================================================

seed_comms, consensus_graph = (
    detect_seed_communities(
        multiplex,
        logger,
        config
    )
)

# ==========================================================
# expansion
# ==========================================================

final_comms = seed_expansion(
    hetero_graph,
    seed_comms,
    config['center_type'],
    logger
)

runtime = time.time() - start_time

# ==========================================================
# evaluation
# ==========================================================

eval_results = evaluate_all(
    consensus_graph,
    hetero_graph,
    final_comms,
    config['center_type'],
    logger
)

from core.metrics import (
    flatten_communities
)

flat_comms = flatten_communities(
    final_comms
)

# if config['center_type']=="author":

author_labels = load_author_labels(
        os.path.join(
            DATASET_DIR,
            'author_label.txt'
        )
    )
author_nmi = calculate_nmi(
        flat_comms,
        author_labels,
        node_prefix='a',
        logger=logger
    )
# else:
paper_labels = load_paper_labels(
        os.path.join(
            DATASET_DIR,
            'paper_label.txt'
        )
    )

paper_nmi = calculate_nmi(
        flat_comms,
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
    flat_comms,
    conference_labels,
    node_prefix='c',
    logger=logger
)

overall_nmi = (
    paper_nmi
    +
    author_nmi
    +
    conference_nmi
) / 3

final_results = {

    'method': 'Hete_MESE',

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

for cid, comm in enumerate(seed_comms):

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

logger.info(
    'Saved communities...'
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