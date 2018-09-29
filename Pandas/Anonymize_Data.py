import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder # other transformer includes MinMaxScaler() StandardScaler(), FunctionTransfomer()

# get rid of warnings
import warnings

'''
Anonymize data by removing identifieable information and 
generate new data based on the variable distributions of original dataset.
'''

df = pd.read_csv("../../Kaggle-Challenge-Pica/Titanic/train.csv")
print(df.shape)

# PassengerId, Name, Ticket, Cabin are personally identifieable information.
# such sensitive information should be anonymous,
# but final dataset should not be too different from the original one and should reflect the initial datasets' distributions.

df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df.dropna(inplace=True)

print(df.shape)

# encode Sex and Embarked with numeric values by LabelEncoder()
# Sex will be coded 0,1, Embarked will be coded 0,1,2

Sex_encoder = LabelEncoder()
Sex_encoder.fit(df["Sex"])
print(list(Sex_encoder.classes_))
encoded_Sex = Sex_encoder.transform(df["Sex"])

Embarked_encoder = LabelEncoder()
Embarked_encoder.fit(df["Embarked"])
print(list(Embarked_encoder.classes_))
encoded_Embarked = Embarked_encoder.transform(df["Embarked"])

encoded_df = pd.DataFrame({"Sex_encoded": encoded_Sex, "Embarked_encoded": encoded_Embarked})
df_encode = pd.concat([df.drop(columns=["Sex", "Embarked"]).reset_index(drop=True), encoded_df], axis="columns")

print(encode_df.shape, df.shape, df_encode.shape)


# determine the best continuous distribution of a variable
def best_fit_distribution(data, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# Anonymizing by sampling from the same distribution
# Core Idea:
# For categorical variable, determine the frequencies of its unique values, and then create a discrete probability distribution with the same frequencies for each unique value.
# For continuous variable, determine the best continuous distribution from a pre-defined list of distributions.

# variable with less than 20 unique values is regarded as categorical variable, 
# and variable with equal to or more than 20 unique values is a continuous one.

UniqValue_num = dict(df_encode.nunique())

categorical_cols, continuous_cols = [], []

for var, uniq_num in UniqValue_num.items():
    if uniq_num < 20:
        categorical_cols.append(var)
    else:
        continuous_cols.append(var)
        
best_distributions = []

for c in continuous_cols:
    best_fit_name, best_fit_params = best_fit_distribution(df_encode[c], 50)
    best_distributions.append((best_fit_name, best_fit_params))

print(best_distributions)



# Generate new dataset based on the variable distributions of original dataset
def generate_df(df, categorical_cols, continuous_cols, best_distributions, n, seed=0):
    np.random.seed(seed)
    new_df = {}

    for c in categorical_cols:
        counts = df[c].value_counts()
        new_df[c] = np.random.choice(list(counts.index), p=(counts/len(df)).values, size=n)

    for c, bd in zip(continuous_cols, best_distributions):
        dist = getattr(scipy.stats, bd[0])
        new_df[c] = dist.rvs(size=n, *bd[1])

    return pd.DataFrame(new_df, columns=categorical_cols+continuous_cols)


simulated_df = generate_df(df_encode, categorical_cols, continuous_cols, best_distributions, n=100)
print(simulated_df.shape)
print(simulated_df.head())

print(simulated_df.nunique())


# One drawback of this approach is that all the interactions between the variables are lost.
# e.g. in the original dataset, women (Sex=1) had a higher chance of surviving (Survived=1) than man (Sex=0). 
# In the generated dataset, this relationship is no longer exsistent. 
# Any other relationship between the variables that might have existed, are lost as well.

