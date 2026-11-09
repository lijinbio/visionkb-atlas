"""Compare majorclass summaries with an earlier per-label pseudo-bulk CPM table.

The earlier tables (``<bname>_mean.txt.gz``: gene x label; ``<bname>_labelsummary.txt``) hold, per label, the mean
over sample x label pseudo-bulks (>= mincell cells) of CPM = 1e6 * count / pseudo-bulk total. Our
``pseudobulk_cp10k`` pools all cells of the label, so values differ in scale and weighting; the comparison is a
Spearman rank correlation across genes per label plus equality of cell counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def crosscheck(summary: pd.DataFrame, mean_table: str, labelsummary: str, level: str = "majorclass") -> pd.DataFrame:
    ref = pd.read_csv(mean_table, sep="\t", index_col=0)
    lab = pd.read_csv(labelsummary, sep="\t")
    key = lab.columns[0]
    s = summary[summary.level == level]
    rows = []
    for label in sorted(set(s.group_label) | set(ref.columns)):
        sub = s[s.group_label == label].set_index("var_symbol")
        rec = {"label": label, "in_summary": len(sub) > 0, "in_reference": label in ref.columns}
        if rec["in_summary"] and rec["in_reference"]:
            common = sub.index.intersection(ref.index)
            a = sub.loc[common, "pseudobulk_cp10k"].to_numpy()
            b = ref.loc[common, label].to_numpy()
            keep = (a > 0) | (b > 0)
            rec["n_genes"] = int(keep.sum())
            rec["spearman_pseudobulk_vs_mean_cpm"] = float(spearmanr(a[keep], b[keep]).statistic)
            rec["spearman_mean_log1p_vs_mean_cpm"] = float(
                spearmanr(sub.loc[common, "mean_log1p_cp10k"].to_numpy()[keep], b[keep]).statistic
            )
            rec["median_ratio_cpm_over_cp10k"] = float(np.median(b[(a > 0) & (b > 0)] / a[(a > 0) & (b > 0)]))
        lr = lab[lab[key] == label]
        if len(lr):
            rec["n_cells_reference"] = int(lr["ncell"].iloc[0])
        if rec["in_summary"]:
            rec["n_cells_summary"] = int(sub["n_cells"].iloc[0])
        rec["n_cells_equal"] = rec.get("n_cells_reference") == rec.get("n_cells_summary")
        rows.append(rec)
    return pd.DataFrame(rows)
