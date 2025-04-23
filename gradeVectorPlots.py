import numpy as np
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver


def _extract_quiver_data(ax):
    """Extract arrays (X, Y, U, V) from each Quiver instance on the Axes."""
    data = []
    for q in ax.findobj(Quiver):
        X = np.array(q.X).flatten()
        Y = np.array(q.Y).flatten()
        U = np.array(q.U).flatten()
        V = np.array(q.V).flatten()
        data.append((X, Y, U, V))
    return data


def _grade_quiver(ax, expected_tails, expected_vectors):
    """
    Compare quiver data on `ax` against expected lists of tails/vectors.
    Returns 100.0 if the multiset of arrows matches, else 0.0, plus {'quiver_count': n}.
    """
    quivers = _extract_quiver_data(ax)
    log = {'quiver_count': len(quivers)}

    # build the expected multiset of (x,y,u,v) tuples
    expected = set(zip(
        [t[0] for t in expected_tails],
        [t[1] for t in expected_tails],
        [v[0] for v in expected_vectors],
        [v[1] for v in expected_vectors],
    ))

    for X, Y, U, V in quivers:
        actual = set(zip(X.flatten(), Y.flatten(), U.flatten(), V.flatten()))
        # if the multisets match exactly, grading is 100%
        if actual == expected:
            return 100.0

    return 0.0


def grade_plot_vector(vectors, tails=None):
    """
    Grade plot_helper.plot_vector by exact arrow match with tiling logic.
    """
    _fig = list(map(plt.figure, plt.get_fignums()))[-1]
    _ax  = _fig.gca()
    vec_arr = np.array(vectors)
    if vec_arr.ndim == 1:
        vec_arr = vec_arr.reshape(1, 2)
    if tails is None:
        tails_arr = np.zeros_like(vec_arr)
    else:
        tails_arr = np.array(tails)
        if tails_arr.ndim == 1:
            tails_arr = tails_arr.reshape(1, 2)
    # tile single vector across multiple tails, or vice versa
    nv = vec_arr.shape[0]
    nt = tails_arr.shape[0]
    if nv == 1 and nt > 1:
        vec_arr   = np.tile(vec_arr,   (nt, 1))
    elif nt == 1 and nv > 1:
        tails_arr = np.tile(tails_arr, (nv, 1))
    else:
        assert vec_arr.shape == tails_arr.shape, "Vectors and tails must align"
    expected_vectors = [tuple(r) for r in vec_arr]
    expected_tails   = [tuple(r) for r in tails_arr]
    return _grade_quiver(_ax, expected_tails, expected_vectors)


def grade_plot_linear_transformation(matrix, *vectors, unit_vector=True, unit_circle=False):
    """
    Grade plot_helper.plot_linear_transformation by exact arrow match.

    Parameters
    ----------
    matrix : array-like of shape (2,2)
        The transformation matrix used in the plot.
    *vectors : sequence of array-like, each of length 2
        The vectors passed to plot_linear_transformation.
    unit_vector : bool, optional
        Whether unit vectors were drawn (default True).
    unit_circle : bool, optional
        Irrelevant for quiver grading (default False).

    Returns
    -------
    float
        100.0 if the plotted arrows exactly match the expected transformed vectors
        (including basis if unit_vector=True), otherwise 0.0.
    """
    # Get the most recent figure and its axes
    fig = list(map(plt.figure, plt.get_fignums()))[-1]
    axes = fig.get_axes()
    if len(axes) < 2:
        raise ValueError("Expected two subplots for before/after transformation")
    # After-transformation axes is the second subplot
    ax2 = axes[1]

    # Build expected tails and vectors
    expected_tails = []
    expected_vectors = []
    mat = np.array(matrix)

    # Include transformed basis vectors if requested
    if unit_vector:
        # First basis: e1 -> mat[:,0]
        I = mat[:, 0]
        # Second basis: e2 -> mat[:,1]
        J = mat[:, 1]
        expected_tails.extend([(0, 0), (0, 0)])
        expected_vectors.extend([(I[0], I[1]), (J[0], J[1])])

    # Include each transformed input vector
    for vec in vectors:
        v = np.array(vec)
        vt = mat @ v.reshape(2, 1)
        expected_tails.append((0, 0))
        expected_vectors.append((float(vt[0, 0]), float(vt[1, 0])))

    # Extract all quiver arrows from the after-transformation axes
    quivers = _extract_quiver_data(ax2)
    actual = set()
    for X, Y, U, V in quivers:
        actual |= set(zip(X.flatten(), Y.flatten(), U.flatten(), V.flatten()))

    # Build the expected set of arrow tuples
    expected = set(zip(
        [t[0] for t in expected_tails],
        [t[1] for t in expected_tails],
        [u for u, _ in expected_vectors],
        [v for _, v in expected_vectors],
    ))

    return 100.0 if actual == expected else 0.0