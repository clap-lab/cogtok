
import numpy as np
from scipy.stats import t, norm
from math import atanh, pow

import pandas as pd
import itertools

# The first two functions have been written by Philipp Singer and are copied from his github repository: https://github.com/psinger/CorrelationStats/
def rz_ci(r, n, conf_level = 0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return np.tanh((zl, zu))

def independent_corr(xy, ab, n, n2 = None, twotailed=True, conf_level=0.95, method='fisher'):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == 'fisher':
        xy_z = 0.5 * np.log((1 + xy)/(1 - xy))
        ab_z = 0.5 * np.log((1 + ab)/(1 - ab))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = (1 - norm.cdf(z))
        if twotailed:
            p *= 2

        return z, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')



# Get sample sizes

sample_sizes ={}
language_map = {"en": "English", "es":"Spanish", "fr": "French", "nl": "Dutch"}
resultpath = "../../results/"
languages = ["en", "es", "fr", "nl"]
for lang in languages:
    evaldata = pd.read_csv(resultpath + "results_overview/" + lang + ".csv")
    evaldata = evaldata.dropna()
    words = evaldata[evaldata["lexicality"]=="W"]
    nonwords = evaldata[evaldata["lexicality"]== "N"]
    sample_sizes[language_map[lang]] = {"Words":len(words), "Non-Words":len(nonwords)}


# Calculate significance for pairwise comparison between models in Figure 1
results = pd.read_csv("subresults/chunkability.csv")

# Uncomment to calculate significance for pairwise comparison between models in Figure 2
#results = pd.read_csv("subresults/pretrained_multilingual_vocab.csv")
for language in set(results["Language"]):
    print("\n--------------------\n")
    print(language)
    language_results = results[results["Language"] == language]

    for category in set(results["Category"]):
        print(category)
        category_results = language_results[language_results["Category"] == category]
        sample_size = sample_sizes[language][category]
        print(sample_size)
        for metric in set(results["Metric"]):
            print(metric)
            metric_results = category_results[category_results["Metric"] == metric]

            correlations = dict(zip(metric_results["Model"], metric_results["Correlation"]))

            for model_a, model_b in itertools.combinations(correlations,2):
                z, p = independent_corr(correlations[model_a], correlations[model_b],  28730)
                # if p <= 0.05:
                #     print("difference significant at 0.05", model_a, model_b)
                # else:
                #     print("NOT significant at 0.05", model_a, model_b)
                if p <= 0.01:
                    print("difference significant at 0.01", model_a, model_b)
                else:
                    print("NOT significant at 0.01", model_a, model_b)
        print()






