from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import scipy.stats as st
from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova"


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata: dict):
        """
        Exaflow version of two-way ANOVA, matching the old exaflow ANOVA:

        - Two categorical predictors x1, x2
        - sstype: 1 or 2 (Type I / Type II SS)
        - Terms: [x1, x2, x1:x2, Residuals]
        """
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput(
                "ANOVA two-way requires exactly one y and at least two x variables."
            )

        xs = self.inputdata.x
        if len(xs) != 2:
            msg = "Anova two-way only works with two independent variables. "
            msg += f"Got {len(xs)} variable(s) instead."
            raise BadUserInput(msg)

        x1, x2 = xs
        y = self.inputdata.y[0]

        sstype = self.parameters.get("sstype")
        if sstype not in (1, 2):
            raise BadUserInput("Parameter 'sstype' must be 1 or 2.")

        # Basic metadata check: at least 2 levels per factor
        x1_enums_meta = list(metadata[x1]["enumerations"])
        x2_enums_meta = list(metadata[x2]["enumerations"])

        if len(x1_enums_meta) < 2:
            raise BadUserInput(
                f"The variable {x1} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )
        if len(x2_enums_meta) < 2:
            raise BadUserInput(
                f"The variable {x2} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )

        # Collect actually observed levels across workers
        level_results = self.engine.run_algorithm_udf(
            func=anova_twoway_collect_levels,
            positional_args={
                "inputdata": self.inputdata.json(),
                "x1": x1,
                "x2": x2,
            },
        )

        levels_a = set()
        levels_b = set()
        for res in level_results:
            levels_a.update(val for val in res["x1_levels"] if val is not None)
            levels_b.update(val for val in res["x2_levels"] if val is not None)

        levels_a = sorted(levels_a)
        levels_b = sorted(levels_b)

        if len(levels_a) < 2:
            raise BadUserInput(
                f"The data of variable {x1} contain less than 2 levels and Anova "
                "cannot be performed. Please select more data or choose another "
                "variable."
            )
        if len(levels_b) < 2:
            raise BadUserInput(
                f"The data of variable {x2} contain less than 2 levels and Anova "
                "cannot be performed. Please select more data or choose another "
                "variable."
            )

        # Run distributed ANOVA in a single UDF with aggregation server
        udf_results = self.engine.run_algorithm_udf(
            func=anova_twoway_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "x1": x1,
                "x2": x2,
                "y": y,
                "levels_a": levels_a,
                "levels_b": levels_b,
                "sstype": sstype,
            },
        )

        metrics: Dict = udf_results[0]

        return AnovaResult(
            terms=metrics["terms"],
            sum_sq=metrics["sum_sq"],
            df=metrics["df"],
            f_stat=metrics["f_stat"],
            f_pvalue=metrics["f_pvalue"],
        )


# ---------------------------------------------------------------------------
# Helper UDFs
# ---------------------------------------------------------------------------


@exaflow_udf()
def anova_twoway_collect_levels(inputdata, x1, x2):
    """
    Collect observed levels for x1 and x2 on each worker.
    """
    import pandas as pd

    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, dropna=True)
    if data.empty:
        return {"x1_levels": [], "x2_levels": []}

    col1 = data[x1]
    col2 = data[x2]

    # If we see DataFrame columns, take the first column
    if isinstance(col1, pd.DataFrame):
        col1 = col1.iloc[:, 0]
    if isinstance(col2, pd.DataFrame):
        col2 = col2.iloc[:, 0]

    levels1 = col1.dropna().unique().tolist()
    levels2 = col2.dropna().unique().tolist()
    return {"x1_levels": levels1, "x2_levels": levels2}


@exaflow_udf(with_aggregation_server=True)
def anova_twoway_local_step(
    inputdata,
    agg_client,
    x1,
    x2,
    y,
    levels_a,
    levels_b,
    sstype,
):
    """
    Distributed two-way ANOVA via linear model approach, with df derived
    from matrix ranks to match the original exaflow implementation.
    """
    import pandas as pd

    from exaflow.algorithms.exaflow.data_loading import load_algorithm_dataframe

    data = load_algorithm_dataframe(inputdata, dropna=True)

    sstype = int(sstype)
    levels_a = list(levels_a)
    levels_b = list(levels_b)
    La = len(levels_a)
    Lb = len(levels_b)

    # If no usable data locally, contribute zeros with correct shapes
    if data.empty or any(col not in data.columns for col in (x1, x2, y)):
        # parameter counts per model
        p_const = 1
        p_a = 1 + (La - 1)
        p_b = 1 + (Lb - 1)
        p_ab = 1 + (La - 1) + (Lb - 1)
        p_full = 1 + (La - 1) + (Lb - 1) + (La - 1) * (Lb - 1)

        zeros_const = np.zeros((p_const, p_const), dtype=float)
        zeros_a = np.zeros((p_a, p_a), dtype=float)
        zeros_b = np.zeros((p_b, p_b), dtype=float)
        zeros_ab = np.zeros((p_ab, p_ab), dtype=float)
        zeros_full = np.zeros((p_full, p_full), dtype=float)

        zeros_const_vec = np.zeros((p_const, 1), dtype=float)
        zeros_a_vec = np.zeros((p_a, 1), dtype=float)
        zeros_b_vec = np.zeros((p_b, 1), dtype=float)
        zeros_ab_vec = np.zeros((p_ab, 1), dtype=float)
        zeros_full_vec = np.zeros((p_full, 1), dtype=float)

        n_local = 0.0
        yTy_local = 0.0

        n_total = int(agg_client.sum([n_local])[0])
        yTy = float(agg_client.sum([yTy_local])[0])

        xTx_const = np.asarray(
            agg_client.sum(zeros_const.ravel().tolist()), dtype=float
        ).reshape(zeros_const.shape)
        xTy_const = np.asarray(
            agg_client.sum(zeros_const_vec.ravel().tolist()), dtype=float
        ).reshape(zeros_const_vec.shape)

        xTx_a = np.asarray(
            agg_client.sum(zeros_a.ravel().tolist()), dtype=float
        ).reshape(zeros_a.shape)
        xTy_a = np.asarray(
            agg_client.sum(zeros_a_vec.ravel().tolist()), dtype=float
        ).reshape(zeros_a_vec.shape)

        xTx_b = np.asarray(
            agg_client.sum(zeros_b.ravel().tolist()), dtype=float
        ).reshape(zeros_b.shape)
        xTy_b = np.asarray(
            agg_client.sum(zeros_b_vec.ravel().tolist()), dtype=float
        ).reshape(zeros_b_vec.shape)

        xTx_ab = np.asarray(
            agg_client.sum(zeros_ab.ravel().tolist()), dtype=float
        ).reshape(zeros_ab.shape)
        xTy_ab = np.asarray(
            agg_client.sum(zeros_ab_vec.ravel().tolist()), dtype=float
        ).reshape(zeros_ab_vec.shape)

        xTx_full = np.asarray(
            agg_client.sum(zeros_full.ravel().tolist()), dtype=float
        ).reshape(zeros_full.shape)
        xTy_full = np.asarray(
            agg_client.sum(zeros_full_vec.ravel().tolist()), dtype=float
        ).reshape(zeros_full_vec.shape)

    else:
        # --- Local design construction ---
        col_y = data[y]
        col_a = data[x1]
        col_b = data[x2]

        if isinstance(col_y, pd.DataFrame):
            col_y = col_y.iloc[:, 0]
        if isinstance(col_a, pd.DataFrame):
            col_a = col_a.iloc[:, 0]
        if isinstance(col_b, pd.DataFrame):
            col_b = col_b.iloc[:, 0]

        col_y = col_y.reset_index(drop=True).astype(float)
        col_a = col_a.reset_index(drop=True)
        col_b = col_b.reset_index(drop=True)

        y_vec = col_y.to_numpy().reshape(-1, 1)
        n_local = float(y_vec.shape[0])

        def encode_factor(values, levels):
            arr_list = []
            for lvl in levels[1:]:
                arr_list.append((values == lvl).astype(float).to_numpy().reshape(-1, 1))
            if not arr_list:
                return np.empty((values.shape[0], 0), dtype=float)
            return np.hstack(arr_list)

        A = encode_factor(col_a, levels_a)  # (n, La-1)
        B = encode_factor(col_b, levels_b)  # (n, Lb-1)

        n_rows = y_vec.shape[0]
        ones = np.ones((n_rows, 1), dtype=float)

        # parameter counts
        p_const = 1
        p_a = 1 + (La - 1)
        p_b = 1 + (Lb - 1)
        p_ab = 1 + (La - 1) + (Lb - 1)
        p_full = 1 + (La - 1) + (Lb - 1) + (La - 1) * (Lb - 1)

        X_const = ones
        X_a = np.hstack([ones, A])
        X_b = np.hstack([ones, B])
        X_ab = np.hstack([ones, A, B])

        inter_cols = []
        if A.shape[1] > 0 and B.shape[1] > 0:
            for j in range(A.shape[1]):
                for k in range(B.shape[1]):
                    inter_cols.append((A[:, j] * B[:, k]).reshape(-1, 1))
        X_full = np.hstack([X_ab] + inter_cols) if inter_cols else X_ab

        assert X_const.shape[1] == p_const
        assert X_a.shape[1] == p_a
        assert X_b.shape[1] == p_b
        assert X_ab.shape[1] == p_ab
        assert X_full.shape[1] == p_full

        # local X'X, X'y, y'y
        xTx_const_local = X_const.T @ X_const
        xTy_const_local = X_const.T @ y_vec

        xTx_a_local = X_a.T @ X_a
        xTy_a_local = X_a.T @ y_vec

        xTx_b_local = X_b.T @ X_b
        xTy_b_local = X_b.T @ y_vec

        xTx_ab_local = X_ab.T @ X_ab
        xTy_ab_local = X_ab.T @ y_vec

        xTx_full_local = X_full.T @ X_full
        xTy_full_local = X_full.T @ y_vec

        yTy_local = float((y_vec**2).sum())

        # --- secure aggregation ---
        n_total = int(agg_client.sum([n_local])[0])
        yTy = float(agg_client.sum([yTy_local])[0])

        xTx_const = np.asarray(
            agg_client.sum(xTx_const_local.ravel().tolist()), dtype=float
        ).reshape(xTx_const_local.shape)
        xTy_const = np.asarray(
            agg_client.sum(xTy_const_local.ravel().tolist()), dtype=float
        ).reshape(xTy_const_local.shape)

        xTx_a = np.asarray(
            agg_client.sum(xTx_a_local.ravel().tolist()), dtype=float
        ).reshape(xTx_a_local.shape)
        xTy_a = np.asarray(
            agg_client.sum(xTy_a_local.ravel().tolist()), dtype=float
        ).reshape(xTy_a_local.shape)

        xTx_b = np.asarray(
            agg_client.sum(xTx_b_local.ravel().tolist()), dtype=float
        ).reshape(xTx_b_local.shape)
        xTy_b = np.asarray(
            agg_client.sum(xTy_b_local.ravel().tolist()), dtype=float
        ).reshape(xTy_b_local.shape)

        xTx_ab = np.asarray(
            agg_client.sum(xTx_ab_local.ravel().tolist()), dtype=float
        ).reshape(xTx_ab_local.shape)
        xTy_ab = np.asarray(
            agg_client.sum(xTy_ab_local.ravel().tolist()), dtype=float
        ).reshape(xTy_ab_local.shape)

        xTx_full = np.asarray(
            agg_client.sum(xTx_full_local.ravel().tolist()), dtype=float
        ).reshape(xTx_full_local.shape)
        xTy_full = np.asarray(
            agg_client.sum(xTy_full_local.ravel().tolist()), dtype=float
        ).reshape(xTy_full_local.shape)

    # ------------------------------------------------------------------
    # Global side: compute RSS for each model
    # ------------------------------------------------------------------
    if n_total == 0:
        raise BadUserInput("ANOVA cannot be performed: no data available.")

    def rss_from_xtx_xty(xtx, xty, yty):
        if xtx.size == 0:
            return float(yty)
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ xty

        # Make sure we go from (1,1) array -> 0-d array -> Python float
        bxty = float((beta.T @ xty).squeeze())
        bxtxb = float((beta.T @ xtx @ beta).squeeze())

        rss = float(yty) - 2.0 * bxty + bxtxb
        return rss

    rss_const = rss_from_xtx_xty(xTx_const, xTy_const, yTy)
    rss_a = rss_from_xtx_xty(xTx_a, xTy_a, yTy)
    rss_b = rss_from_xtx_xty(xTx_b, xTy_b, yTy)
    rss_ab = rss_from_xtx_xty(xTx_ab, xTy_ab, yTy)
    rss_full = rss_from_xtx_xty(xTx_full, xTy_full, yTy)

    # ------------------------------------------------------------------
    # Degrees of freedom from matrix ranks (to match exaflow)
    # ------------------------------------------------------------------
    r_const = np.linalg.matrix_rank(xTx_const)
    r_a = np.linalg.matrix_rank(xTx_a)
    r_b = np.linalg.matrix_rank(xTx_b)
    r_ab = np.linalg.matrix_rank(xTx_ab)
    r_full = np.linalg.matrix_rank(xTx_full)

    df_a = max(r_a - r_const, 0)
    df_b = max(r_ab - r_a, 0)
    df_inter = max(r_full - r_ab, 0)
    df_resid = int(n_total - r_full)

    if df_resid <= 0:
        raise BadUserInput(
            "ANOVA cannot be performed: residual degrees of freedom <= 0."
        )

    # ------------------------------------------------------------------
    # Sum of squares by SS type
    # ------------------------------------------------------------------
    sum_sq = np.empty(4, dtype=float)

    if sstype == 1:
        # Type I (sequential): const -> a -> b -> a:b
        sum_sq[0] = rss_const - rss_a  # A (x1)
        sum_sq[1] = rss_a - rss_ab  # B (x2)
        sum_sq[2] = rss_ab - rss_full  # A:B
        sum_sq[3] = rss_full  # Residual
    else:  # sstype == 2
        # Type II: A and B adjusted for the other, interaction as usual
        sum_sq[0] = rss_b - rss_ab  # A | B
        sum_sq[1] = rss_a - rss_ab  # B | A
        sum_sq[2] = rss_ab - rss_full  # A:B
        sum_sq[3] = rss_full  # Residual

    # ------------------------------------------------------------------
    # Interaction SS & df corrections
    # ------------------------------------------------------------------
    # If interaction adds no rank, treat it as df=0 and SS=0
    if df_inter == 0:
        sum_sq[2] = 0.0

    # If interaction SS still negative (numerical issues), recompute from
    # total model SS: SS_total = RSS_const - RSS_full = SS_A + SS_B + SS_AB
    ss_inter_orig = float(sum_sq[2])
    if ss_inter_orig < 0 and df_inter > 0:
        ss_total_model = rss_const - rss_full
        ss_a = float(sum_sq[0])
        ss_b = float(sum_sq[1])
        ss_inter_new = ss_total_model - ss_a - ss_b
        sum_sq[2] = ss_inter_new

    df = np.array([df_a, df_b, df_inter, df_resid], dtype=int)

    # ------------------------------------------------------------------
    # F and p-values
    # ------------------------------------------------------------------
    ms = np.zeros_like(sum_sq)
    # avoid division by zero for df=0
    for i in range(4):
        if df[i] > 0:
            ms[i] = sum_sq[i] / df[i]
        else:
            ms[i] = 0.0

    # F for A, B, interaction; None for residual
    F = [None, None, None, None]
    if df[0] > 0:
        F[0] = ms[0] / ms[3] if ms[3] != 0 else None
    if df[1] > 0:
        F[1] = ms[1] / ms[3] if ms[3] != 0 else None
    if df[2] > 0:
        F[2] = ms[2] / ms[3] if ms[3] != 0 else None

    pval: List[Optional[float]] = [None, None, None, None]
    if F[0] is not None:
        pval[0] = float(1.0 - st.f.cdf(F[0], df[0], df[3]))
    if F[1] is not None:
        pval[1] = float(1.0 - st.f.cdf(F[1], df[1], df[3]))
    if F[2] is not None:
        pval[2] = float(1.0 - st.f.cdf(F[2], df[2], df[3]))

    terms = [x1, x2, f"{x1}:{x2}", "Residuals"]

    return {
        "terms": terms,
        "sum_sq": sum_sq.tolist(),
        "df": df.tolist(),
        "f_stat": F,
        "f_pvalue": pval,
    }
