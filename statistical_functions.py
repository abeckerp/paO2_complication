import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from scipy.stats import norm as snorm
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests
import scikit_posthocs as sp
from typing import Union

def posthoc_dunn(
        a: Union[list, np.ndarray, pd.DataFrame],
        val_col: str = None,
        group_col: str = None,
        p_adjust: str = None,
        sort: bool = True) -> pd.DataFrame:
    '''Post hoc pairwise test for multiple comparisons of mean rank sums
    (Dunn's test). May be used after Kruskal-Wallis one-way analysis of
    variance by ranks to do pairwise comparisons [1]_, [2]_.

    Parameters
    ----------
    a : array_like or pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.

    val_col : str, optional
        Name of a DataFrame column that contains dependent variable values (test
        or response variable). Values should have a non-nominal scale. Must be
        specified if `a` is a pandas DataFrame object.

    group_col : str, optional
        Name of a DataFrame column that contains independent variable values
        (grouping or predictor variable). Values should have a nominal scale
        (categorical). Must be specified if `a` is a pandas DataFrame object.

    p_adjust : str, optional
        Method for adjusting p values. See `statsmodels.sandbox.stats.multicomp`
        for details. Available methods are:
        'bonferroni' : one-step correction
        'sidak' : one-step correction
        'holm-sidak' : step-down method using Sidak adjustments
        'holm' : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel' : closed method based on Simes tests (non-negative)
        'fdr_bh' : Benjamini/Hochberg  (non-negative)
        'fdr_by' : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (non-negative)
        'fdr_tsbky' : two stage fdr correction (non-negative)

    sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    Returns
    -------
    result : pandas.DataFrame
        P values.

    Notes
    -----
    A tie correction will be employed according to Glantz (2012).

    References
    ----------
    .. [1] O.J. Dunn (1964). Multiple comparisons using rank sums.
        Technometrics, 6, 241-252.
    .. [2] S.A. Glantz (2012), Primer of Biostatistics. New York: McGraw Hill.

    Examples
    --------

    >>> x = [[1,2,3,5,1], [12,31,54, np.nan], [10,12,6,74,11]]
    >>> sp.posthoc_dunn(x, p_adjust = 'holm')
    '''
    def compare_dunn(i, j):
        diff = np.abs(x_ranks_avg.loc[i] - x_ranks_avg.loc[j])
        A = n * (n + 1.) / 12.
        B = (1. / x_lens.loc[i] + 1. / x_lens.loc[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2. * snorm.sf(np.abs(z_value))
        return p_value, z_value

    x, _val_col, _group_col = sp.__convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()

    # ties
    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = tie_sum / (12. * (n - 1))

    vs = np.zeros((x_len, x_len))
    z_vals = np.zeros((x_len, x_len))
    combs = combinations(range(x_len), 2)

    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0
    z_vals[:, :] = 0

    for i, j in combs:
        vs[i, j], z_vals[i,j] = compare_dunn(x_groups_unique[i], x_groups_unique[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

    vs[tri_lower] = np.transpose(vs)[tri_lower]
    z_vals[tri_lower] = np.transpose(z_vals)[tri_lower]
    np.fill_diagonal(vs, 1)
    np.fill_diagonal(z_vals, 1)
    return pd.DataFrame(vs, index=x_groups_unique, columns=x_groups_unique), pd.DataFrame(z_vals, index=x_groups_unique, columns=x_groups_unique)


def get_asterisks_for_pval(p_val: float, alpha: float = 0.05) -> str:
    """Receives the p-value and returns asterisks string."""
    if p_val > alpha:  # bigger than alpha
        p_text = "ns"
    # following the standards in biological publications
    elif p_val < 1e-4:  # 0.0001
        p_text = "****"
    elif p_val < 1e-3:  # 0.001
        p_text = "***"
    elif p_val < 1e-2:  # 0.01
        p_text = "**"
    else:
        p_text = "*"  # 0.05

    return p_text  # string of asterisks


def run_chisq_on_combination(df: object, combinations_tuple: list) -> float:
    """Receives a dataframe and a combinations tuple and returns p-value after performing chisq test."""
    assert (
        len(combinations_tuple) == 2
    ), "Combinations tuple is too long! Should be of size 2."
    new_df = df[
        (df.index == combinations_tuple[0]) | (df.index == combinations_tuple[1])
    ]
    chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
    return p


def chisq_and_posthoc_corrected(
    df: object, correction_method: str = "fdr_bh", alpha: float = 0.05
) -> None:
    """Receives a dataframe and performs chi2 test and then post hoc.
    Prints the p-values and corrected p-values (after FDR correction).
    alpha: optional threshold for rejection (default: 0.05)
    correction_method: method used for mutiple comparisons correction. (default: 'fdr_bh').
    See statsmodels.sandbox.stats.multicomp.multipletests for elaboration."""

    # start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(df, correction=True)
    print("Chi2 result of the contingency table: {}, p-value: {}\n".format(chi2, p))

    # post-hoc test
    all_combinations = list(
        combinations(df.index, 2)
    )  # gathering all combinations for post-hoc chi2
    print("Post-hoc chi2 tests results:")
    p_vals = [
        run_chisq_on_combination(df, comb) for comb in all_combinations
    ]  # a list of all p-values
    # the list is in the same order of all_combinations

    # correction for multiple testing
    reject_list, corrected_p_vals = multipletests(
        p_vals, method=correction_method, alpha=alpha
    )[:2]
    for p_val, corr_p_val, reject, comb in zip(
        p_vals, corrected_p_vals, reject_list, all_combinations
    ):
        print(
            "{}: p_value: {:5f}; corrected: {:5f} ({}) reject: {}".format(
                comb, p_val, corr_p_val, get_asterisks_for_pval(p_val, alpha), reject
            )
        )


def get_p_values(*args: list) -> float:
    obs = np.array([*args])
    # print("normal - mild - moderate - severe")
    print(obs)
    p = chi2_contingency(obs, correction=False)[1]
    print("Chi2 Ergebnis correction False:", p)
    chisq_and_posthoc_corrected(pd.DataFrame(obs))
    return p


def return_table(df: object, var_list: list, dimensions: int = None) -> list:
    if dimensions == None:
        dimensions = len(var_list)

    if dimensions > 1:
        print(
            f"Table that is converted to an array:\n{df.groupby(var_list).size().unstack(fill_value=0).stack()}"
        )
        np_list = np.array(df.groupby(var_list).size().unstack(fill_value=0).stack())

    else:
        print(f"Table that is converted to an array:\n{df.groupby(var_list).size()}")
        np_list = np.array(df.groupby(var_list).size())

    result = [np_list[i : i + dimensions] for i in range(0, len(np_list), dimensions)]
    print(f"Splitted set that is returned based on {dimensions} dimensions:\n{result}")
    return result


def return_dict(df: object, var_list: list, dimensions: int = None) -> list:
    if dimensions == None:
        dimensions = len(var_list)

    if dimensions > 1:
        print(
            f"Table that is converted to an dictionary and returned:\n{df.groupby(var_list).size().unstack(fill_value=0).stack()}"
        )
        return df.groupby(var_list).size().unstack(fill_value=0).stack().to_dict()
    print(
        f"Table that is converted to an dictionary and returned:\n{df.groupby(var_list).size()}"
    )
    return df.groupby(var_list).size().to_dict()


def compare_frequencies(
    d: dict, exp: list = None, transpose: bool = True, pr: bool = True
) -> float:
    vals, cols, inx = [], [], []
    pair = []
    if exp is None:
        for idx, (key, value) in enumerate(d.items()):
            if value < 5:
                print(
                    f"Attention: This observation ({value}) is <5, thus the test may not be valid!"
                )
            if idx % len({k[1] for k in d.keys()}) == 0:
                if len(pair) > 0:
                    # print(f"Pair being added to values: {pair}")
                    vals.append(pair)
                pair = [value]
            else:
                pair.append(value)

            if idx == len(d.items()) - 1:
                # print(f"Last pair being added to values: {pair}")
                vals.append(pair)
            #                 if expected is not None:
            #                     vals.append(expected)
            #                     inx.append('base')

            if key[1] not in cols:
                cols.append(key[1])
            if key[0] not in inx:
                inx.append(key[0])
    else:
        for idx, (key, value) in enumerate(d.items()):
            if idx % len({k[1] for k in d.keys()}) == 0:
                if len(pair) > 0:
                    # print(f"Pair being added to values: {pair}")
                    vals.append(pair)
                pair = [value]
            else:
                pair.append(value)

            if idx == len(d.items()) - 1:
                # print(f"Last pair being added to values: {pair}")
                vals.append(pair)
                vals.append(exp)
            #                 inx.append('base')

            if key not in cols:
                cols.append(key)
            #             if idx not in inx:
            #                 inx.append(idx)
            inx = [r for r in range(len(vals))]

    if pr == True:
        print(f"Values: {vals}\ncolumns: {cols}\nindices: {inx}")
    tmp_df = pd.DataFrame(vals, columns=cols, index=inx)

    if transpose == True:
        tmp_df = tmp_df.T

    if "_none" in tmp_df.index:
        tmp_df = tmp_df.drop(index="_none")

    ### check for rows with 0
    tmp_df_t = tmp_df.T
    for c in tmp_df_t.columns:
        if tmp_df_t[c].sum() == 0:
            tmp_df_t = tmp_df_t.drop(columns=[c])
    tmp_df = tmp_df_t.T

    observed = np.array(tmp_df)
    if pr:
        print(f"Observed values: \n{tmp_df}")
    chi2, p, dof, expected = chi2_contingency(observed, correction=False)
    if pr:
        print(f"#################\n#####Chi2 p-value: {p}\n#################")
    all_combinations = list(combinations(tmp_df.index, 2))
    # if p<0.05 and len(all_combinations)!=len(observed):
    if p < 0.05:
        alpha = 0.05
        correction_method = "fdr_bh"
        print(f"Observed values: \n{tmp_df}")
        print("Post-hoc chi2 tests results:")
        p_vals = [
            chi2_contingency(
                tmp_df[(tmp_df.index == comb[0]) | (tmp_df.index == comb[1])],
                correction=False,
            )[1]
            for comb in all_combinations
        ]
        reject_list, corrected_p_vals = multipletests(
            p_vals, method=correction_method, alpha=alpha
        )[:2]
        for p_val, corr_p_val, reject, comb in zip(
            p_vals, corrected_p_vals, reject_list, all_combinations
        ):
            print(
                f"{comb}:\t p_value: {round(p_val,4)}; corrected: {round(corr_p_val,4)} ({get_asterisks_for_pval(p_val, alpha)})  \treject: {reject}"
            )

    return p

def unique_cross_product(list1, list2):
    result = []
    seen = set()
    
    for a in list1:
        for b in list2:
            if a != b:  # keine Paare mit gleichen Zahlen
                pair = tuple(sorted((a, b)))  # (1,3) und (3,1) -> (1,3)
                if pair not in seen:
                    seen.add(pair)
                    result.append(pair)
    
    return result

def effect_size(dataframe, agg, vals, row_indices, col_indices):
    # calculate effect size independently of p values
    agg_vals = [x[0] for x in list(dataframe.groupby(agg))] # how many distinct values exist for the aggregation/groupby
    for (row_index, col_index) in unique_cross_product(row_indices, col_indices):
    # for i in range(len(row_indices)):
    #     row_index = row_indices[i]
    #     col_index = col_indices[i]

        n1 = len(dataframe.loc[dataframe[agg] == agg_vals[row_index], vals])
        s1 = np.std(dataframe.loc[dataframe[agg] == agg_vals[row_index], vals])
        n2 = len(dataframe.loc[dataframe[agg] == agg_vals[col_index], vals])
        s2 = np.std(dataframe.loc[dataframe[agg] == agg_vals[col_index], vals])

        # z = zvalues.loc[row_index+1, col_index+1]
        # r = np.abs(z/np.sqrt(n1+n2))

        # if r <= 0.1:
        #     r_text = "none"
        # elif r <= 0.3:
        #     r_text = "weak"
        # elif r <= 0.5:
        #     r_text = "moderate"
        # elif r > 0.5:
        #     r_text = "strong"
        # print(f"\n{row_index+1}, {col_index+1}: Effect size (Cohen, 1992): {round(r,2)} ({r_text})")

        # Hedge's g
        pooled_s = np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
        m1 = np.mean(dataframe.loc[dataframe[agg] == agg_vals[row_index], vals])
        m2 = np.mean(dataframe.loc[dataframe[agg] == agg_vals[col_index], vals])
        d = np.abs(m1-m2)/pooled_s
        if d <= 0.01:
            d_text = "none"
        elif d <= 0.2:
            d_text = "very small"
        elif d <= 0.5:
            d_text = "small"
        elif d <= 0.8:
            d_text = "moderate"
        elif d <= 1.2:
            d_text = "large"
        elif d <= 2.0:
            d_text = "very large"
        else:
            d_text = "huge"
        if d>=0.01:
            print(f"\n{row_index+1}, {col_index+1}: Effect size: Hedges' g: {round(d,2)} ({d_text})")
        else:
            print(f"\n{row_index+1}, {col_index+1}: Hedges' g <0.01")
        # return d

def posthoc_median(vals: list, dataframe: object, agg: str, p: float) -> None:
    ph_dunn = 1
    if p < 0.05:
        x = [group[vals].values for name, group in dataframe.groupby(agg)]
        ph_dunn, zvalues = posthoc_dunn(x, p_adjust="fdr_bh", sort=True)

        if ((ph_dunn < 0.05).any()).any():
            print(sp.sign_table(ph_dunn))
            pd.options.display.float_format = '{:.4f}'.format
            print(ph_dunn)

            ## effect size
            row_indices, col_indices = np.where(
                pd.DataFrame(sp.sign_table(ph_dunn)).isin(["*", "**", "***"])
            )
            effect_size(dataframe, agg, vals, row_indices, col_indices)
    elif ph_dunn >= 0.05:
        print("No significance found.")
        ## effect size
        agg_vals = [x[0] for x in list(dataframe.groupby(agg))]
        effect_size(dataframe, agg, vals, list(range(len(agg_vals))), list(range(len(agg_vals))))


def posthoc_anova(vals: list, dataframe: object, agg: str) -> None:
    x = [group[vals].values for name, group in dataframe.groupby(agg)]
    ph_ttest = sp.posthoc_ttest(x, p_adjust="fdr_bh", sort=True)
    if ((ph_ttest < 0.05).any()).any():
        print(sp.sign_table(ph_ttest))
        print(ph_ttest)
    else:
        print("No significance found.")


def compare_median(vals: list, dataframe: object, agg: str, pr: bool = True) -> float:
    if pr:
        print(dataframe.groupby([agg]).agg({vals: ["median", "mean", "count"]}))
    try:
        z, p = kruskal(
            *[
                group[vals].values
                for name, group in dataframe.groupby(agg)
                if len(group) >= 5
            ]
        )
    except:
        if pr:
            print("Need at least two groups...")
        p = 1
    if pr:
        print(f"p-value: {round(p,4)}")
    if p < 0.05:
        if not pr: # print only if not printed before
            print(dataframe.groupby([agg]).agg({vals: ["median", "mean", "count"]}))
        print("\n... Post hoc test...")
    posthoc_median(vals, dataframe, agg, p)
    return p


def compare_variance(vals, dataframe, agg, pr=True):
    if pr:
        print(dataframe.groupby([agg]).agg({vals: ["median", "mean", "count"]}))
    try:
        p = scipy.stats.f_oneway(
            *[
                group[vals].values
                for name, group in dataframe.groupby(agg)
                if len(group) >= 2
            ]
        )[1]
    except:
        print("Need at least two groups...")
        p = 1
    if pr:
        print(f"p-value: {round(p,4)}")
    if p < 0.05:
        print(dataframe.groupby([agg]).agg({vals: ["median", "mean", "count"]}))
        print("... Post hoc test...")
        posthoc_anova(vals, dataframe, agg)
    return p
