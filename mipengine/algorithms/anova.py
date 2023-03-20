from typing import List
from typing import Optional

import numpy as np
import scipy.stats as st
from pydantic import BaseModel

from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.linear_regression import LinearRegression
from mipengine.algorithms.preprocessing import FormulaTransformer
from mipengine.algorithms.preprocessing import relation_to_vector
from mipengine.exceptions import BadUserInput


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWay(Algorithm, algname="anova"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="Two-way analysis of variance (ANOVA)",
            label="Two-way ANOVA",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="independent",
                    desc="independent variable",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                ),
                y=InputDataSpecification(
                    label="dependent",
                    desc="Dependent variable",
                    types=[InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "sstype": ParameterSpecification(
                    label="sstype",
                    desc="Type of sum of squares to use. It can be 1 or 2.",
                    types=[ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    default="2",
                    min=1,
                    max=2,
                ),
            },
        )

    def get_variable_groups(self):
        [y] = self.variables.y
        x1, x2 = self.variables.x
        return [[y], [x1, x2]]

    def run(self, data_model_views, metadata):
        [[y], [x1, x2]] = self.get_variable_groups()

        Y, X = data_model_views

        x1_enums = list(metadata[x1]["enumerations"])
        x2_enums = list(metadata[x2]["enumerations"])
        sstype = self.algorithm_parameters["sstype"]

        if len(x1_enums) < 2:
            raise BadUserInput(
                f"The variable {x1} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )
        if len(x2_enums) < 2:
            raise BadUserInput(
                f"The variable {x2} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )

        # Handy formula aliases
        const = "1"
        a = x1
        b = x2
        a_b = f"{a} + {b}"
        a_b_ab = f"{a} * {b}"

        # ANOVA is computed using a linear model computational approach.
        # A number of lms are defined based on the SS type, and each model
        # is fitted to the data.
        if sstype == 1:
            formulas = [const, a, b, a_b, a_b_ab]
        elif sstype == 2:
            formulas = [a, b, a_b, a_b_ab]

        # Define datasets for each lm based on above formulas
        transformers = {
            formula: FormulaTransformer(self.engine, self.variables, metadata, formula)
            for formula in formulas
        }
        Xs = {
            formula: transformer.transform(X)
            for formula, transformer in transformers.items()
        }

        # Once transform runs we can gen the actual enums
        x1_enums = transformers[a_b_ab].enums[a]
        x2_enums = transformers[a_b_ab].enums[b]
        if len(x1_enums) < 2:
            raise BadUserInput(
                f"The data of variable {x1} contain less than 2 levels and Anova "
                "cannot be performed. Please select more data or choose another "
                "variable."
            )
        if len(x2_enums) < 2:
            raise BadUserInput(
                f"The data of variable {x2} contain less than 2 levels and Anova "
                "cannot be performed. Please select more data or choose another "
                "variable."
            )

        # Define lms and fit to data
        models = {formula: LinearRegression(self.engine) for formula in formulas}
        for formula in formulas:
            X = Xs[formula]
            model = models[formula]
            model.fit(X, Y)
            model.compute_summary(
                y_test=relation_to_vector(Y, self.engine),
                y_pred=model.predict(X),
                p=len(X.columns) - 1,
            )

        # ANOVA computation
        terms = [x1, x2, f"{x1}:{x2}", "Residuals"]

        sum_sq = np.empty(4)
        if sstype == 1:
            sum_sq[0] = models[const].rss - models[a].rss
            sum_sq[1] = models[a].rss - models[a_b].rss
            sum_sq[2] = models[a_b].rss - models[a_b_ab].rss
            sum_sq[3] = models[a_b_ab].rss
        elif sstype == 2:
            sum_sq[0] = models[b].rss - models[a_b].rss
            sum_sq[1] = models[a].rss - models[a_b].rss
            sum_sq[2] = models[a_b].rss - models[a_b_ab].rss
            sum_sq[3] = models[a_b_ab].rss

        df = compute_df(transformers[a_b_ab].design_info, models[a_b_ab])

        ms = sum_sq / df
        F = [ms[0] / ms[3], ms[1] / ms[3], ms[2] / ms[3], None]

        pval = [None, None, None, None]
        pval[0] = 1 - st.f.cdf(F[0], df[0], df[3])
        pval[1] = 1 - st.f.cdf(F[1], df[1], df[3])
        pval[2] = 1 - st.f.cdf(F[2], df[2], df[3])

        return AnovaResult(
            terms=terms,
            sum_sq=sum_sq.tolist(),
            df=df.tolist(),
            f_stat=F,
            f_pvalue=pval,
        )


def compute_df(design_info, model):
    # The code for computing dfs is copied from statsmodels
    arr = np.zeros((len(design_info.terms), len(design_info.column_names)))
    term_names, slices = zip(*design_info.term_name_slices.items())

    for i, slice_ in enumerate(slices):
        arr[i, slice_] = 1

    term_names = np.array(term_names)
    idx = term_names == "Intercept"

    df_no_model = arr[~idx].sum(1)

    df = np.zeros(len(df_no_model) + 1, dtype=int)
    df[:-1] = df_no_model
    df[-1] = model.df

    return df
