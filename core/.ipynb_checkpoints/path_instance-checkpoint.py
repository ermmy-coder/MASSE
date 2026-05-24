"""
Deprecated:
DFS meta-path search.

Replaced by:
MetaPathEngine sparse matrix framework.
"""
from collections import defaultdict

TYPE_MAPPING = {
    'writes': 'P',
    'be_written': 'A',
    'published_in': 'C',
    'publish': 'P',
    'has_term': 'T',
    'term_in': 'P'
}

def find_path_instances(
    u,
    v,
    meta_path,
    graph,
    logger=None
):

    path_types = meta_path.split('-')

    instances = []

    def dfs(current, path, remain):

        if len(remain) == 0:

            if current == v:
                instances.append(path)

            return

        rel_type = remain[0]

        for edge in graph['edges']:

            src, dst, etype = edge

            if src != current:
                continue

            mapped = TYPE_MAPPING.get(etype)

            if mapped == rel_type:

                dfs(
                    dst,
                    path + [edge],
                    remain[1:]
                )

    dfs(u, [], path_types[1:])

    if logger:

        logger.debug(
            f'[META_PATH] {u}->{v} '
            f'{meta_path} '
            f'instances={len(instances)}'
        )

    return instances