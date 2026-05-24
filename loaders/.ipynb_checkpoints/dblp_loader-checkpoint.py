import os
import json


def read_node_file(
    filepath,
    prefix
):

    nodes = []

    with open(
        filepath,
        'r',
        encoding='latin1',
        errors='ignore'
    ) as f:

        for line in f:

            line = line.strip()

            if line == '':
                continue

            parts = line.split()

            raw_id = parts[0]

            nodes.append(
                f'{prefix}{raw_id}'
            )

    return nodes


def read_edge_file(
    filepath,
    src_prefix,
    dst_prefix,
    rel_type,
    reverse_rel=None
):

    edges = []

    with open(
        filepath,
        'r',
        encoding='latin1'
    ) as f:

        for line in f:

            line = line.strip()

            if line == '':
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            src = int(parts[0])
            dst = int(parts[1])

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


def load_dblp_dataset(dataset_dir):

    hetero_graph = {
        'author': [],
        'paper': [],
        'conference': [],
        'term': [],
        'edges': []
    }

    print('\n========== LOADING DBLP ==========')

    # ======================================================
    # 节点
    # ======================================================

    hetero_graph['author'] = read_node_file(
        os.path.join(dataset_dir, 'author.txt'),
        'a'
    )

    hetero_graph['paper'] = read_node_file(
        os.path.join(dataset_dir, 'paper.txt'),
        'p'
    )

    hetero_graph['conference'] = read_node_file(
        os.path.join(dataset_dir, 'conf.txt'),
        'c'
    )

    hetero_graph['term'] = read_node_file(
        os.path.join(dataset_dir, 'term.txt'),
        't'
    )

    print(f'Authors: {len(hetero_graph["author"])}')
    print(f'Papers: {len(hetero_graph["paper"])}')
    print(f'Conferences: {len(hetero_graph["conference"])}')
    print(f'Terms: {len(hetero_graph["term"])}')

    # ======================================================
    # Paper-Author
    # ======================================================

    pa_edges = read_edge_file(
        os.path.join(dataset_dir, 'paper_author.txt'),
        'p',
        'a',
        'writes',
        'be_written'
    )

    hetero_graph['edges'].extend(pa_edges)

    print(f'Paper-Author edges: {len(pa_edges)}')

    # ======================================================
    # Paper-Conference
    # ======================================================

    pc_edges = read_edge_file(
        os.path.join(dataset_dir, 'paper_conf.txt'),
        'p',
        'c',
        'published_in',
        'publish'
    )

    hetero_graph['edges'].extend(pc_edges)

    print(f'Paper-Conference edges: {len(pc_edges)}')

    # ======================================================
    # Paper-Term
    # ======================================================

    pt_edges = read_edge_file(
        os.path.join(dataset_dir, 'paper_term.txt'),
        'p',
        't',
        'has_term',
        'term_in'
    )

    hetero_graph['edges'].extend(pt_edges)

    print(f'Paper-Term edges: {len(pt_edges)}')

    print(
        f'\nTotal edges: '
        f'{len(hetero_graph["edges"])}'
    )

    return hetero_graph


def save_processed_graph(
    hetero_graph,
    save_path
):

    with open(save_path, 'w') as f:

        json.dump(
            hetero_graph,
            f
        )

    print(f'\nSaved graph to {save_path}')


if __name__ == "__main__":

    dataset_dir = '../datasets/DBLP/'

    graph = load_dblp_dataset(
        dataset_dir
    )

    os.makedirs(
        '../datasets/DBLP/processed',
        exist_ok=True
    )

    save_processed_graph(
        graph,
        '../datasets/DBLP/processed/hetero_graph.json'
    )