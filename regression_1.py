"""A regression on the wine dataset using pandas's OLS model"""
# pylint: disable=C0103
import pandas as pd
import statsmodels.formula.api as sm

# Load the data
df = pd.read_csv('./data/winequality-red.csv', sep=';')
print df[:1]

# Run a simple OLS
result = sm.ols(
    formula='''
    quality ~ alcohol
    ''', data=df
).fit()

print result.summary()

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                quality   R-squared:                       0.227
# Model:                            OLS   Adj. R-squared:                  0.226
# Method:                 Least Squares   F-statistic:                     468.3
# Date:                Thu, 27 Jul 2017   Prob (F-statistic):           2.83e-91
# Time:                        09:44:25   Log-Likelihood:                -1721.1
# No. Observations:                1599   AIC:                             3446.
# Df Residuals:                    1597   BIC:                             3457.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      1.8750      0.175     10.732      0.000       1.532       2.218
# alcohol        0.3608      0.017     21.639      0.000       0.328       0.394
# ==============================================================================
# Omnibus:                       38.501   Durbin-Watson:                   1.748
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               71.758
# Skew:                          -0.154   Prob(JB):                     2.62e-16
# Kurtosis:                       3.991   Cond. No.                         104.
# ==============================================================================
#
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# Run a simple OLS
# result = sm.ols(
#     formula='''
#     quality ~ fixed_acidity + volatile_acidity + citric_acid
#     + residual_sugar + chlorides + free_sulfur_dioxide + density
#     + pH + sulphates + alcohol
#     ''', data=df
# ).fit()
#
# print result.summary()

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                quality   R-squared:                       0.352
# Model:                            OLS   Adj. R-squared:                  0.348
# Method:                 Least Squares   F-statistic:                     86.44
# Date:                Thu, 27 Jul 2017   Prob (F-statistic):          3.63e-142
# Time:                        01:36:39   Log-Likelihood:                -1579.2
# No. Observations:                1599   AIC:                             3180.
# Df Residuals:                    1588   BIC:                             3240.
# Df Model:                          10
# Covariance Type:            nonrobust
# =======================================================================================
#                           coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              28.7507     21.267      1.352      0.177     -12.964      70.465
# fixed_acidity           0.0519      0.025      2.045      0.041       0.002       0.102
# volatile_acidity       -1.1997      0.119    -10.082      0.000      -1.433      -0.966
# citric_acid            -0.3573      0.143     -2.502      0.012      -0.637      -0.077
# residual_sugar          0.0139      0.015      0.921      0.357      -0.016       0.043
# chlorides              -1.6109      0.418     -3.857      0.000      -2.430      -0.792
# free_sulfur_dioxide    -0.0021      0.002     -1.276      0.202      -0.005       0.001
# density               -25.5162     21.695     -1.176      0.240     -68.070      17.038
# pH                     -0.2472      0.189     -1.307      0.191      -0.618       0.124
# sulphates               0.8994      0.115      7.824      0.000       0.674       1.125
# alcohol                 0.2861      0.027     10.776      0.000       0.234       0.338
# ==============================================================================
# Omnibus:                       24.508   Durbin-Watson:                   1.749
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               36.999
# Skew:                          -0.144   Prob(JB):                     9.24e-09
# Kurtosis:                       3.687   Cond. No.                     4.23e+04
# ==============================================================================
#
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 4.23e+04. This might indicate that there are
# strong multicollinearity or other numerical problems.
# [Finished in 1.065s]
