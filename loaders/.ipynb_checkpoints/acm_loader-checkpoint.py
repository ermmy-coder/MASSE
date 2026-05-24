import os
import json

import scipy.io as sio
from scipy.sparse import csr_matrix


# ==========================================================
# build node list
# ==========================================================

def build_nodes(
    num_nodes,
    prefix
):

    nodes = []

    for i in range(num_nodes):

        nodes.append(
            f'{prefix}{i}'
        )

    return nodes


# ==========================================================
# build edges from sparse matrix
# ==========================================================

def build_edges_from_matrix(
    matrix,
    src_prefix,
    dst_prefix,
    rel_type,
    reverse_rel=None
):

    edges = []

    matrix = csr_matrix(matrix)

    rows, cols = matrix.nonzero()

    for src, dst in zip(rows, cols):

        edges.append(
            (
                f'{src_prefix}{src}',
                f'{dst_prefix}{dst}',
                rel_type
            )
        )

        if reverse_rel is not None:

            edges.append(
                (
                    f'{dst_prefix}{dst}',
                    f'{src_prefix}{src}',
                    reverse_rel
                )
            )

    return edges


# ==========================================================
# load ACM dataset
# ==========================================================

def load_acm_dataset(dataset_dir):

    hetero_graph = {

        'paper': [],

        'author': [],

        'subject': [],

        'conference': [],

        'edges': []
    }

    print('\n========== LOADING ACM ==========')

    mat_path = os.path.join(
        dataset_dir,
        'ACM.mat'
    )

    data = sio.loadmat(mat_path)

    print('\n========== MAT KEYS ==========')

    for k in data.keys():

        print(k)

    # ======================================================
    # relation matrices
    # ======================================================

    PvsA = data['PvsA']

    PvsL = data['PvsL']

    PvsC = data['PvsC']

    # ======================================================
    # nodes
    # ======================================================

    hetero_graph['paper'] = build_nodes(
        PvsA.shape[0],
        'p'
    )

    hetero_graph['author'] = build_nodes(
        PvsA.shape[1],
        'a'
    )

    hetero_graph['subject'] = build_nodes(
        PvsL.shape[1],
        's'
    )

    hetero_graph['conference'] = build_nodes(
        PvsC.shape[1],
        'c'
    )

    print(
        f'Papers: '
        f'{len(hetero_graph["paper"])}'
    )

    print(
        f'Authors: '
        f'{len(hetero_graph["author"])}'
    )

    print(
        f'Subjects: '
        f'{len(hetero_graph["subject"])}'
    )

    print(
        f'Conferences: '
        f'{len(hetero_graph["conference"])}'
    )

    # ======================================================
    # Paper-Author
    # ======================================================

    pa_edges = build_edges_from_matrix(

        PvsA,

        'p',
        'a',

        'writes',

        'be_written'
    )

    hetero_graph['edges'].extend(
        pa_edges
    )

    print(
        f'Paper-Author edges: '
        f'{len(pa_edges)}'
    )

    # ======================================================
    # Paper-Subject
    # ======================================================

    ps_edges = build_edges_from_matrix(

        PvsL,

        'p',
        's',

        'has_subject',

        'subject_of'
    )

    hetero_graph['edges'].extend(
        ps_edges
    )

    print(
        f'Paper-Subject edges: '
        f'{len(ps_edges)}'
    )

    # ======================================================
    # Paper-Conference
    # ======================================================

    pc_edges = build_edges_from_matrix(

        PvsC,

        'p',
        'c',

        'published_in',

        'publish'
    )

    hetero_graph['edges'].extend(
        pc_edges
    )

    print(
        f'Paper-Conference edges: '
        f'{len(pc_edges)}'
    )

    print(
        f'\nTotal edges: '
        f'{len(hetero_graph["edges"])}'
    )
    print('\n========== LABEL DEBUG ==========')

    print(type(data['T']))
    
    print(data['T'].shape)

    return hetero_graph


# ==========================================================
# save processed graph
# ==========================================================

def save_processed_graph(
    hetero_graph,
    save_path
):

    with open(save_path, 'w') as f:

        json.dump(
            hetero_graph,
            f
        )

    print(
        f'\nSaved graph to '
        f'{save_path}'
    )


# ==========================================================
# main
# ==========================================================

if __name__ == "__main__":

    dataset_dir = '../datasets/ACM/'

    graph = load_acm_dataset(
        dataset_dir
    )

    os.makedirs(
        '../datasets/ACM/processed',
        exist_ok=True
    )

    save_processed_graph(

        graph,

        '../datasets/ACM/processed/hetero_graph.json'
    )