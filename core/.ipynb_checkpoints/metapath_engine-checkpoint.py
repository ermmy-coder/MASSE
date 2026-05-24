import numpy as np
import scipy.sparse as sp
from collections import defaultdict


class MetaPathEngine:

    def __init__(
        self,
        hetero_graph,
        logger=None
    ):

        self.graph = hetero_graph

        self.logger = logger

        self.node_maps = {}

        self.reverse_maps = {}

        self.adj_matrices = {}

        self._build_node_indices()

        self._build_relation_matrices()

    # ======================================================
    # 节点编号
    # ======================================================

    def _build_node_indices(self):

        for node_type, nodes in self.graph.items():

            if node_type == 'edges':
                continue

            self.node_maps[node_type] = {
                node: idx
                for idx, node in enumerate(nodes)
            }

            self.reverse_maps[node_type] = {
                idx: node
                for idx, node in enumerate(nodes)
            }

    # ======================================================
    # 构建 relation adjacency matrix
    # ======================================================

    def _build_relation_matrices(self):

        relation_edges = defaultdict(list)

        for u, v, rel in self.graph['edges']:

            relation_edges[rel].append((u, v))

        relation_node_types = {

            # ==========================================
            # paper-author
            # ==========================================
        
            'writes': (
                'paper',
                'author'
            ),
        
            'be_written': (
                'author',
                'paper'
            ),
        
            # ==========================================
            # paper-conference
            # ==========================================
        
            'published_in': (
                'paper',
                'conference'
            ),
        
            'publish': (
                'conference',
                'paper'
            ),
        
            # ==========================================
            # paper-term
            # ==========================================
        
            'has_term': (
                'paper',
                'term'
            ),
        
            'term_in': (
                'term',
                'paper'
            )
        }

        for rel, edges in relation_edges.items():

            if rel not in relation_node_types:
                continue

            src_type, dst_type = relation_node_types[rel]

            src_size = len(
                self.graph[src_type]
            )

            dst_size = len(
                self.graph[dst_type]
            )

            rows = []
            cols = []

            for u, v in edges:

                rows.append(
                    self.node_maps[src_type][u]
                )

                cols.append(
                    self.node_maps[dst_type][v]
                )

            data = np.ones(len(rows))

            mat = sp.csr_matrix(
                (
                    data,
                    (rows, cols)
                ),
                shape=(
                    src_size,
                    dst_size
                )
            )

            self.adj_matrices[rel] = mat

            if self.logger:

                self.logger.info(
                    f'[MATRIX] '
                    f'{rel} shape={mat.shape}'
                )

    # ======================================================
    # Meta-path multiplication
    # ======================================================

    def compute_meta_path(
        self,
        relations
    ):

        """
        relations:
        ['writes',
         'has_term',
         'term_in',
         'be_written']
        """

        current = self.adj_matrices[
            relations[0]
        ]

        for rel in relations[1:]:

            current = current.dot(
                self.adj_matrices[rel]
            )

        return current

    # ======================================================
    # matrix -> weighted graph
    # ======================================================

    def matrix_to_edges(
        self,
        matrix,
        src_type,
        threshold=1
    ):

        coo = matrix.tocoo()

        edges = []

        for i, j, val in zip(
            coo.row,
            coo.col,
            coo.data
        ):

            if i == j:
                continue

            if val < threshold:
                continue

            u = self.reverse_maps[
                src_type
            ][i]

            v = self.reverse_maps[
                src_type
            ][j]

            edges.append(
                (u, v, val)
            )

        return edges