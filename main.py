import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions as sm_except
from statsmodels.regression.rolling import RollingOLS
from scipy import stats

filename = 'data/InterviewTestData.xlsx'
outlier_detect_value = 4 #In standard deviations
# Section A: Load and Clean data
try:
    port_returns = pd.read_excel(filename, skiprows=1, sheet_name='input_portfolio').dropna(axis=1, how='all')
    factor_returns = pd.read_excel(filename, skiprows=1, sheet_name='input_factors').dropna(axis=1, how='all')
except FileNotFoundError as nf:
    raise nf

#Arrange Data
port_returns['Date'] = pd.to_datetime(port_returns['Date'])
port_returns = port_returns.set_index(keys='Date')
factor_returns['Date'] = pd.to_datetime(factor_returns['Date'])
factor_returns = factor_returns.set_index(keys='Date')

#Check data
try:
    port_returns = port_returns.astype('float')
    factor_returns = factor_returns.astype('float')
except ValueError:
    raise

#Replace string. Needs a more robust cleaning exercise here but will
# do for now
port_returns = port_returns.replace('---', np.NaN)
port_returns = port_returns.astype('float')

# Ffill as per instructions, perhaps 0 might be better
factor_returns = factor_returns.ffill()
port_returns = port_returns.ffill()

benchmark = 'MSCI World'
factors = factor_returns.columns.drop(benchmark)
portfolios = port_returns.columns.drop(benchmark)

# Summary Statistics
port_returns.describe()
factor_returns.describe()

#Item 1: Outlier Detection

port_returns.plot(kind='box')
factor_returns.plot(kind='box')

# Z-Scores
factor_zscore = (factor_returns - factor_returns.mean()) / factor_returns.std()
port_zscore = (port_returns - port_returns.mean()) / port_returns.std()

ax1 = sns.heatmap(factor_zscore)

y_dates = factor_zscore.index.strftime('%Y-%m-%d')
ax1.set_yticklabels(labels=y_dates)

plt.show()

ax2 = sns.heatmap(port_zscore)
y_dates = port_zscore.index.strftime('%Y-%m-%d')
ax2.set_yticklabels(labels=y_dates)

plt.show()


# If one assumes a normal distribution then a 4 sigma move is extremely unlikely,
# #3sigma covers ~ 99.99%. Given the data this is reasonable for now. We would need to check
# the veracity of these moves. We set these returns to 0, assuming bad data. Winsorisation would
# be another possibility, and replace these outliers with average etc.
outlier_mask = np.abs(factor_zscore) > outlier_detect_value
factor_outliers = factor_returns[outlier_mask].dropna(axis=0, how='all').dropna(axis=1, how='all')
factor_returns.iloc[outlier_mask] = 0


outlier_mask = np.abs(port_zscore) > outlier_detect_value
port_outliers = port_returns[outlier_mask].dropna(axis=0, how='all').dropna(axis=1, how='all')
port_returns.iloc[outlier_mask] = 0

port_returns.plot(kind='box')
factor_returns.plot(kind='box')

# Item 2: Portfolio Excess Returns
port_ex_returns = port_returns[portfolios].subtract(port_returns[benchmark], axis=0)

# Item 3: Cumulative Factor Returns
cumulative_port_return = (1 + factor_returns).cumprod() - 1
# Start at 100
cumulative_port_return = 100 + cumulative_port_return
#Remove Benchmark as we are interested in Factors only
cumulative_port_return[factors].plot()

#Item 4: Yearly Excess Return, Annualised and Not Annualised
port_ex_returns["Year"] = port_ex_returns.index.strftime("%Y")
yearly_returns = port_ex_returns.groupby(by='Year', axis=0).agg(lambda x: (1+x).prod()-1)
yearly_returns_annualised = port_ex_returns.groupby(by='Year', axis=0).agg(lambda x: pow((1+x).prod(), 12/len(x))) - 1

#Item 5: Factor Excess Returns
factor_ex_returns = factor_returns[factors].subtract(factor_returns[benchmark], axis=0)

#Item 6: OLS

joined_data = port_returns[portfolios].join(factor_returns)
X = joined_data[factors]
X = sm.add_constant(X)
for portfolio in portfolios:
    y = joined_data[portfolio]
    model = sm.OLS(y, X)
    y_hat = model.fit()
    print(y_hat.summary())
    print('R2: ', y_hat.rsquared)
    print("Params: ", y_hat.params)

    model = RollingOLS(joined_data[portfolios[1]], X, window=36, min_nobs=36)
    rolling_res = model.fit()
    params = rolling_res.params
    rolling_res.pvalues
    rolling_res.mse_resid.plot()

# Item 7:
cor = port_ex_returns[portfolios].corr()
print(cor)
mask = np.triu(np.ones_like(cor, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


last_year_cor = port_ex_returns.tail(n=12)
cor = last_year_cor[portfolios].corr()
print(cor)
mask = np.triu(np.ones_like(cor, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

