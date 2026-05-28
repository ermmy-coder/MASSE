from cdlib import algorithms


# ==========================================================
# Louvain
# ==========================================================

def run_louvain(G):

    comms = algorithms.louvain(
        G
    ).communities

    return [
        set(c)
        for c in comms
    ]



# ==========================================================
# Label Propagation
# ==========================================================

def run_lpa(G):

    comms = algorithms.label_propagation(
        G
    ).communities

    return [
        set(c)
        for c in comms
    ]


# ==========================================================
# Infomap
# ==========================================================

def run_infomap(G):

    comms = algorithms.infomap(
        G
    ).communities

    return [
        set(c)
        for c in comms
    ]


# ==========================================================
# Leiden
# ==========================================================
def run_leiden(
    G,
    resolution=1.0,
    weights=None
):

    comms = algorithms.leiden(

        G,

        weights=weights,

        # resolution_parameter=resolution
    ).communities

    return [
        set(c)
        for c in comms
    ]