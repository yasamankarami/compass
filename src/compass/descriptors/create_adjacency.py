import numpy as np

def edge_fused_adjacency(mat_dict, coupling_keys, mindist_key=None,
                         contact_cutoff=None, var_keep=0.90,
                         return_mode="scalar", out_path=None, prec=4):
    """
    Direct-coupling fusion at the EDGE level (no profile-similarity).
      scalar   -> (n,n) [0,1] adjacency, oriented PC1  (drop-in for COMPASS)
      features -> (n,n,k) edge-feature stack, top-k PCs to var_keep  (for a GNN)
    Missing residues (entirely-NaN row/col) carry no edges.
    """
    feats = [np.asarray(mat_dict[k]["data"], float) for k in coupling_keys]
    n = feats[0].shape[0]
    assert all(f.shape == (n, n) for f in feats), "shape mismatch"
    miss = np.zeros(n, bool)
    for f in feats:
        miss |= np.isnan(f).all(1) | np.isnan(f).all(0)
    valid = ~miss
    iu = np.triu_indices(n, 1)
    ok = valid[iu[0]] & valid[iu[1]]
    if mindist_key is not None and contact_cutoff is not None:
        MIN = np.asarray(mat_dict[mindist_key]["data"], float)
        ok &= (MIN[iu] < contact_cutoff)
    ii, jj = iu[0][ok], iu[1][ok]
    E = np.column_stack([f[ii, jj] for f in feats])
    assert np.isfinite(E).all(), "stray NaN in edge features"
    Z = (E - E.mean(0)) / (E.std(0) + 1e-12); Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    evr = (S**2) / (S**2).sum()
    if return_mode == "scalar":
        s = Zc @ Vt[0]
        if np.corrcoef(s, Z.mean(1))[0, 1] < 0: s = -s
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        A = np.full((n, n), np.nan); A[ii, jj] = s; A[jj, ii] = s
        np.fill_diagonal(A, 1.0)
        if out_path: np.savetxt(out_path, np.nan_to_num(A), fmt=f"%.{prec}f")
        return A, evr
    k = int(np.searchsorted(np.cumsum(evr), var_keep) + 1)
    sc = Zc @ Vt[:k].T
    F = np.full((n, n, k), np.nan)
    for c in range(k):
        F[ii, jj, c] = sc[:, c]; F[jj, ii, c] = sc[:, c]
    if out_path: np.savetxt(out_path, np.nan_to_num(F), fmt=f"%.{prec}f")
    return F, evr[:k]
