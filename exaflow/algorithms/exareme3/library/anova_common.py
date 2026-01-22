from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd


def get_min_max_ci_info(
    *,
    means: List[float],
    sample_stds: List[float],
    categories: List[str],
    var_min_per_group: List[float],
    var_max_per_group: List[float],
    group_stats_index: List[str],
) -> Tuple[Dict, Dict]:
    """Builds min/max per group and confidence interval information tables."""
    categories = [c for c in categories if c in group_stats_index]
    df1_means_stds_dict = {
        "categories": categories,
        "sample_stds": list(sample_stds),
        "means": list(means),
    }
    df_min_max = {
        "categories": categories,
        "min": var_min_per_group,
        "max": var_max_per_group,
    }
    df1_means_stds = pd.DataFrame(df1_means_stds_dict, index=categories).drop(
        "categories", axis=1
    )
    df1_means_stds["m-s"] = df1_means_stds["means"] - df1_means_stds["sample_stds"]
    df1_means_stds["m+s"] = df1_means_stds["means"] + df1_means_stds["sample_stds"]

    min_max_per_group = df_min_max
    ci_info = df1_means_stds.to_dict()

    return min_max_per_group, ci_info
