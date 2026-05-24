import os
import sys
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
    load_paper_labels,
    calculate_nmi
)

from utils.visualize import (
    visualize_communities
)

from utils.graph_utils import (
    hetero_to_networkx
)

# ==========================================================
# logger
# ==========================================================

logger = get_logger(
    'hete_masse',
    'debug_logs/dblp.log'
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
    'DBLP'
)

print(f'[INFO] Dataset dir: {DATASET_DIR}')

# ==========================================================
# load dataset
# ==========================================================

hetero_graph = load_dblp_dataset(
    DATASET_DIR
)

# ==========================================================
# paper-centric meta-paths
# ==========================================================

meta_paths = {

    'P-A-P': 1.0,

    'P-T-P': 0.8,

    'P-C-P': 0.6
}

# ==========================================================
# start timing
# ==========================================================

start_time = time.time()

# ==========================================================
# build multiplex network
# ==========================================================

multiplex = build_multiplex_network(
    hetero_graph,
    'paper',
    meta_paths,
    logger
)

# ==========================================================
# consensus communities
# ==========================================================

seed_comms = detect_seed_communities(
    multiplex,
    logger
)

# ==========================================================
# stable expansion
# ==========================================================

final_comms = seed_expansion(
    hetero_graph,
    seed_comms,
    'paper',
    logger
)

# ==========================================================
# runtime
# ==========================================================

runtime = time.time() - start_time

logger.info(
    f'[RUNTIME] {runtime:.2f}s'
)

# ==========================================================
# evaluation
# ==========================================================

logger.info(
    '\n========== EVALUATION =========='
)

eval_results = evaluate_all(
    hetero_graph,
    seed_comms,
    logger
)

# ==========================================================
# NMI
# ==========================================================

label_path = os.path.join(
    DATASET_DIR,
    'paper_label.txt'
)

labels = load_paper_labels(
    label_path
)

nmi = calculate_nmi(
    seed_comms,
    labels,
    logger
)

# ==========================================================
# final report
# ==========================================================

final_results = {

    'modularity':
        eval_results['modularity'],

    'surprise':
        eval_results['surprise'],

    'keyword_coherence':
        eval_results['keyword_coherence'],

    'nmi':
        nmi,

    'runtime_seconds':
        runtime,

    'community_statistics':
        eval_results['statistics']
}

# ==========================================================
# print final results
# ==========================================================

print('\n')
print('=' * 60)
print('FINAL RESULTS')
print('=' * 60)

for k, v in final_results.items():

    print(f'{k}: {v}')

print('=' * 60)

# ==========================================================
# save results
# ==========================================================

os.makedirs(
    'results',
    exist_ok=True
)

with open(
    'results/dblp_results.json',
    'w'
) as f:

    def convert_numpy(obj):

        if hasattr(obj, 'item'):
    
            return obj.item()
    
        raise TypeError
    
    
    json.dump(
        final_results,
        f,
        indent=4,
        default=convert_numpy
    )

logger.info(
    'Saved results to '
    'results/dblp_results.json'
)

# ==========================================================
# save communities
# ==========================================================

community_save = []

for cid, comm in enumerate(seed_comms):

    community_save.append({

        'community_id': cid,

        'nodes': list(comm)
    })

with open(
    'results/dblp_communities.json',
    'w'
) as f:

    json.dump(
        community_save,
        f,
        indent=4
    )

logger.info(
    'Saved communities to '
    'results/dblp_communities.json'
)

# ==========================================================
# visualization
# ==========================================================

logger.info(
    'Generating visualization...'
)

G = hetero_to_networkx(
    hetero_graph
)

visualize_communities(
    G,
    seed_comms,
    'DBLP Communities',
    'results/dblp_communities.png'
)

logger.info(
    'Visualization saved'
)

print('\nExperiment Finished.')