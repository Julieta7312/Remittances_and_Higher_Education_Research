# import the packages
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.cross_decomposition import PLSRegression
from statsmodels.tools.tools import pinv_extended
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sys import displayhook
import seaborn as sns
import pandas as pd
import numpy as np

# File paths for the variables' raw data
enr_file = '../data/tertiary_school_enrollment.csv'
exp_file = '../data/household_expenditure.csv'
uem_file = '../data/youth_unem_rate.csv'
rem_file = '../data/inward_remittance_flows.xlsx'
pop_file = '../data/total_world_population.csv'
ppp_file = '../data/ppp_per_capita.csv'

''' Create a list with the targetted Post-Soviet countries to filter the dataframe with those countries only '''
ps_ctry = ['Armenia', 'Azerbaijan', 'Georgia', 'Kyrgyz Republic', 'Uzbekistan', 'Tajikistan', 'Belarus', 'Moldova', 'Ukraine', 'Latvia'] 
processed_from_date = '2002/1/1'

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: READ & TIDY UP THE DATA *********************************''')

''' data1. Annual tertiary school enrollment for all the countries (last updated: 2021/06/30) 
        : World Bank tertiary school enrollment data (% gross) '''

enr = pd.read_csv(enr_file, sep=',')
enr.columns = enr.iloc[3]
enr = enr.iloc[4:] \
         .drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
enr.columns = list(enr.columns[:1]) + [ pd.to_datetime(int(year), format='%Y') for year in enr.columns[1:] ]
enr = enr.rename(columns={'Country Name' : 'Country'}) \
         .set_index('Country').T

''' data1.1. Subset - School enrollment (Post-Soviet countries)
        : from year 2000 onwards the enrolment dataframe has 42 total missing values '''

ps_enr = enr[ps_ctry].reset_index().melt(id_vars=['index'])
ps_enr.columns = ['Date', 'Country', 'enr_rate']
ps_enr['enr_rate'] = ps_enr['enr_rate'].apply(lambda x : float(x) if x!=".." else np.nan) / 100

''' data2. Household expenditure for the Post-Soviet countries as a depedent variable (in USD) '''

exp = pd.read_csv(exp_file, sep = ',').T
exp.columns = exp.iloc[0,:]
exp.index.name = "Date"
exp = exp.iloc[1:]
exp.index = pd.to_datetime(exp.index, format='%Y')
exp = exp.reset_index().melt(id_vars='Date', value_name='hhld_exp $', var_name='Country')
exp['hhld_exp $'] = exp['hhld_exp $'].astype(float)
exp = exp[exp["Country"].isin(ps_ctry)]

''' data3. Unemployment '''

unem = pd.read_csv(uem_file, sep = ',')
unem.columns = list(unem.columns[:4]) + [pd.to_datetime(year, format='%Y') for year in unem.columns[4:]]
unem = unem[ list([unem.columns[0]]) + list(unem.columns[4:]) ].melt(id_vars='Country Name')
unem.columns = ['Country', 'Date', 'unem_rate']
unem['unem_rate'] = unem['unem_rate'] / 100

''' data4. Annual remittances (Global) for all the countries (last updated: 2021/5/30)
        : Migrant remittance annual inflows (US$ million) received by the world's countries from year 1960 onwards '''

remit = pd.read_excel(rem_file)
remit = remit.rename(columns = {'Migrant remittance inflows (US$ million)' : 'Date', '2020e':'2020'})
remit = remit[ [ remit.columns[0] ] + [ col for col in remit.columns[21:-1] ] ]
remit.columns = [ remit.columns[0] ] + [ pd.to_datetime(year, format='%Y') for year in remit.columns[1:] ]

''' data4.1. Subset - Remittances (Post-Soviet countries, in Million USD)
        : from year 2000 onwards the enrolment dataframe has 16 total missing values '''

ps_remit = remit[remit['Date'].apply(lambda c : c in ps_ctry)].melt(id_vars='Date', value_name='rem $', var_name='Dates')
ps_remit['rem $'] = ps_remit['rem $'] * 1000000
ps_remit.columns = ['Country', 'Date', 'rem $']

''' data5. Population (World)
        : Obtain the population of the Post-Soviet coutries to calculate their GDP per capita '''

pop = pd.read_csv(pop_file)
pop.columns = pop.iloc[3]
pop = pop.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
pop.columns = list(pop.columns[:1]) + [ pd.to_datetime(round(year), format='%Y') for year in pop.columns[1:] ]
pop = pop.iloc[4:].rename(columns={'Country Name' : 'Country'})
pop = pop.set_index('Country').T 

''' data5.1. Subset - Population (Post-Soviet countries) '''  

ps_pop = pop[ps_ctry].reset_index().melt(id_vars=['index'])
ps_pop.columns = ['Date', 'Country', 'pop']

''' data6. PPP (only for Post-Soviet countries)
        : Obtain the PPP of the Post-Soviet countries to use as an independent variable '''

ppp = pd.read_csv(ppp_file).melt(id_vars=['Country Name'], value_name = 'PPP_pc $', var_name = 'Date')
ppp['Date'] = ppp['Date'].apply(lambda y: pd.to_datetime(y, format='%Y'))
ppp = ppp[ppp['Country Name'].isin(ps_ctry)]
ppp.columns = ['Country', 'Date', 'PPP_pc $']

''' Merge & sort the data 
        every dataframes are len=168, from 2000-20
        remittance column - 16 null values
        enrollment column - 41 null values '''

for i, df in enumerate( [ ps_enr, exp, unem, ps_remit, ps_pop, ppp ] ):
    if i==0: panel_df = df.copy()
    else: panel_df = pd.merge(left=panel_df, right=df, how='inner', left_on=['Date','Country'], right_on=['Date', 'Country'])

panel_df = panel_df.sort_values(by=['Country', 'Date'])
panel_df.dropna().groupby("Country")["Date"].apply(lambda x : print("THE LIST OF DATES _________________ :", list(x)[0]) )

print('''*********************************** END: READ & TIDY UP THE DATA *********************************''')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: DATA PROCESSING *********************************''')

rem_lag = 1
ppp_lag = 1
uem_lag = 1

''' PPP and remittance '''

# hhld_exp_pc = household expenditure per capita = household expenditure / population
panel_df['hhld_exp_pc $'] = panel_df['hhld_exp $'] / panel_df['pop'] 

# PPP to household expenditure per capita = PPP per capita / household expenditure per capita
panel_df['PPP/hhld_exp'] = panel_df['PPP_pc $']/panel_df['hhld_exp_pc $'] 

# Share of the remittance from PPP = PPP_rem = remittance * PPP / household expenditure 
panel_df['PPP_rem $'] = panel_df['rem $'] * panel_df['PPP/hhld_exp'] 

# rem/GDP
panel_df["rem/GDP"] = panel_df['PPP_rem $'] / (panel_df['PPP_pc $'] * panel_df['pop'])
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
sns.lineplot(data=panel_df, x='Date', y="rem/GDP", hue="Country", palette="icefire")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# unemployment rate plot
sns.lineplot(data=panel_df, x='Date', y="unem_rate", hue="Country", palette="icefire")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# PPP remittance per capita lagged by 1 year
panel_df['PPP_rem_pc_{t-1} $'] = ( panel_df['PPP_rem $'] / panel_df['pop'] ).shift(rem_lag) 

# PPP remittance per capita lagged by 1 year, centered by every country, divided by 1000 to convert to thousands of dollars. 
panel_df['PPP_rem_pc_{t-1} cent,$'] = ( panel_df['PPP_rem_pc_{t-1} $'] - panel_df['PPP_rem_pc_{t-1} $'].mean() ) / 1000 

# PPP per capita - PPP remittance per capita in $=USD
panel_df['( PPP - PPP_rem )_pc $'] = panel_df['PPP_pc $'] - ( panel_df['PPP_rem $'] / panel_df['pop'] )

# PPP per capita - PPP remittance per capita in $=USD lagged by a year
panel_df['( PPP - PPP_rem )_pc_{t-1} $'] = panel_df['( PPP - PPP_rem )_pc $'].shift(ppp_lag)

# PPP per capita - PPP remittance per capita in $=USD lagged by a year, and centralized by all countries, divided by 1000 to convert to thousands of dollars. 
panel_df['( PPP - PPP_rem )_pc_{t-1} cent,$'] = ( panel_df['( PPP - PPP_rem )_pc_{t-1} $'] - panel_df['( PPP - PPP_rem )_pc_{t-1} $'].mean() ) / 1000 


''' Unemployment rate '''

panel_df['unem_rate_{t-1} cent'] = ( panel_df['unem_rate'] - panel_df['unem_rate'].mean() ).shift(uem_lag)


''' Enrollment (dependent variable) '''

# Get the mean of the enrollment rate column
mean_enroll = panel_df.groupby('Country')[['enr_rate']].mean().reset_index()
mean_enroll.columns = ['Country','enr_rate_mean']

# Get the standard deviation of the enrollment rate column
std_enroll = panel_df.groupby('Country')[['enr_rate']].std().reset_index()
std_enroll.columns = ['Country','enr_rate_std']

# Add the mean enrollment rate and the standard deviation of the enrollment rate to the panel_df
panel_df = pd.merge(pd.merge(panel_df, mean_enroll, how='left', left_on='Country', right_on='Country'), std_enroll, how='left', left_on='Country', right_on='Country')

# Subtract the mean enrollment rate from every observation of the enrollment rate column, and divide with the standard deviation of the enrollment rate.
panel_df['enr_rate_standard'] = ( panel_df['enr_rate'] - panel_df['enr_rate_mean'] ) / panel_df['enr_rate_std']


enr_reg_name  = 'enr_rate_standard'
unem_reg_name = 'unem_rate_{t-1} cent'
rem_reg_name  = 'PPP_rem_pc_{t-1} cent,$'
ppp_reg_name  = '( PPP - PPP_rem )_pc_{t-1} cent,$'
reg_var_list  = [enr_reg_name, rem_reg_name, ppp_reg_name, unem_reg_name]
sig = 3 # clip the data if it's more than 3 standard deviation apart from the mean ( lower = mean - 3 standard deviation, upper = mean + standard deviation ). 

''' winsorize '''
for rn in reg_var_list: 
        panel_df[rn] = panel_df[rn].clip( lower = panel_df[rn].mean() - (sig * panel_df[rn].std()), \
                upper = (sig * panel_df[rn].std()) + panel_df[rn].mean() )

panel_df = panel_df[ ['Date', 'Country'] + reg_var_list ]
panel_df = panel_df.query( " Date >= '" + processed_from_date + "'" )

# Generate a dummy variable for each country. 
for ctry in ps_ctry: 
        panel_df[ ctry ] = ( panel_df['Country'] == ctry ).apply( lambda x : int(x) )

panel_df = panel_df.dropna()
panel_df = panel_df.query("not (Date == '2006-01-01' and Country == 'Azerbaijan')")
panel_df = panel_df.query("not (Date == '2010-01-01' and Country == 'Uzbekistan')")

displayhook(pd.concat([panel_df.describe(), pd.DataFrame(panel_df.skew(), columns=["skew"]).T], axis=0).T)


''' PLOTS '''
# Check the existence of a linear relationship between the dependent and independent variables with a scatterplot
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=2)
sns.set_context("paper", font_scale=1.2) 
splot = sns.pairplot(panel_df[reg_var_list+["Country"]], hue="Country", palette="icefire", grid_kws={"despine": False})
plt.show()

# Check the correlation between the variables with a heatmap
sns.set(font_scale=1.4)
htmp = sns.heatmap(panel_df[panel_df.columns[:6]].corr(), vmin=-1, vmax=1, annot=True, cmap="vlag")
plt.show()
''' PLOTS '''


panel_df = panel_df.set_index(['Country', 'Date'])

print('''*********************************** END: DATA PROCESSING *********************************''')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: FIXED EFFECT MODEL, RIDGE REGRESSION, PARTIAL LEAST SQUARES REGRESSION *********************************''')

endg = panel_df[reg_var_list[:1]]
exog = panel_df[reg_var_list[1:] + ps_ctry]
c_type = 'HC1'

endg.to_csv('../data/endg.csv', index=True)
exog.to_csv('../data/exog.csv', index=True)
endg.to_csv('../data/endgNoIdx.csv', index=False)
exog.to_csv('../data/exogNoIdx.csv', index=False)

''' 

Covariance type (c_type variable)
    
    'HC0': White's (1980) heteroskedasticity robust standard errors.
    'HC1', 'HC2', 'HC3': MacKinnon and White's (1985) heteroskedasticity robust standard errors.
    'robust': White”s robust covariance

    // Ridge Regression: The standard error for Beta is computed with empirical covariance of the residuals (can't be used with fixed effect -> clustered covariance is required)
    
'''



print(''' __ START: Pooled OLS Model ____''')

pooled_ols_mdl = sm.OLS(endg, exog[reg_var_list[1:]])
fitted_pooled_ols_mdl = pooled_ols_mdl.fit(cov_type=c_type)
displayhook(fitted_pooled_ols_mdl.summary())
print(''' ____ END: Pooled OLS Model ____''')



print(''' __ START: Pooled OLS Model: Breusch-Pagan-Test ____''')

fitted_pooled_ols_estimations = ( fitted_pooled_ols_mdl.params[:3] * panel_df[reg_var_list[1:]] ).sum(axis=1)
fitted_pooled_ols_residuals = fitted_pooled_ols_mdl.resid

pool_est_resid = pd.DataFrame(fitted_pooled_ols_estimations, columns=["Pooled OLS estimated enrollment rate"]).merge(pd.DataFrame(fitted_pooled_ols_residuals, columns=["residual"]), left_index=True, right_index=True)
pool_est_resid_exog = pd.DataFrame(pool_est_resid).merge(exog, left_index=True, right_index=True)

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)

sns.scatterplot(data=pool_est_resid_exog, x="Pooled OLS estimated enrollment rate", y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among Pooled OLS's fitted values and it's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=rem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the remittance and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=ppp_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the PPP and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=unem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the youth unemployment and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pool_breusch_pagan_test_results = het_breuschpagan(pool_est_resid_exog['residual'], panel_df[reg_var_list[1:]])
displayhook(pd.DataFrame(pool_breusch_pagan_test_results, columns=["Pooled OLS model: Breusch pagan test result"], index=["LM-Stat", "LM p-val", "F-Stat", "F p-val"]))

print(''' ____ END: Pooled OLS Model: Breusch-Pagan-Test ____''')



print(''' __ START: Fixed Effect Model ____''')

fixed_effect_mdl = sm.OLS(endg, exog)
fitted_fixed_effect_mdl = fixed_effect_mdl.fit(cov_type=c_type)
displayhook(fitted_fixed_effect_mdl.summary())

print(''' ____ END: Fixed Effect Model ____''')


print(''' __ START: Fixed Effect Model: Breusch-Pagan-Test ____''')

fitted_fixed_effect_estimations = ( fitted_fixed_effect_mdl.params[:3] * panel_df[reg_var_list[1:]] ).sum(axis=1)
fitted_fixed_effect_residuals = fitted_fixed_effect_mdl.resid

est_resid = pd.DataFrame(fitted_fixed_effect_estimations, columns=["Fixed effect estimated enrollment rate"]).merge(pd.DataFrame(fitted_fixed_effect_residuals, columns=["residual"]), left_index=True, right_index=True)
est_resid_exog = pd.DataFrame(est_resid).merge(exog, left_index=True, right_index=True)

sns.scatterplot(data=est_resid_exog, x="Fixed effect estimated enrollment rate", y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among Fixed effect model's fitted values and it's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=rem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the remittance and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=ppp_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the PPP and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=unem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
# plt.gca().set_title("Heteroscedasticity among the youth unemployment and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

breusch_pagan_test_results = het_breuschpagan(est_resid_exog['residual'], panel_df[reg_var_list[1:]])
displayhook(pd.DataFrame(breusch_pagan_test_results, columns=["Fixed effect model: Breusch pagan test result"], index=["LM-Stat", "LM p-val", "F-Stat", "F p-val"]))

print(''' ____ END: Fixed Effect Model: Breusch-Pagan-Test ____''')



print(''' __ START: Verifying Algorithm ____''')

print("This proves that the model.wexog is equal to X (=A matrix of independent variables, whitened? exogenous variables): ", np.sum(exog.values) == np.sum(fixed_effect_mdl.wexog))

pseudo_inv_X, singular_vals = pinv_extended(fixed_effect_mdl.wexog) # Return the pseudo inverse of X as well as the singular values used in computation.
norm_cov_params = np.dot(pseudo_inv_X, np.transpose(pseudo_inv_X)) # X^{-1} \dot (X^{-1})^T
pseudo_inv_XTX, singular_values = pinv_extended(np.dot(fixed_effect_mdl.wexog.T, fixed_effect_mdl.wexog))
print( "This shows that fixed_effect_mdl.normalized_cov_params = X^{-1} \dot (X^{-1})^T = ( X^T \dot X )^{-1}: ", np.sum(norm_cov_params) - np.sum(pseudo_inv_XTX) < 1e-13 )

summary = sm.regression.linear_model.OLSResults(fixed_effect_mdl, fitted_fixed_effect_mdl.params, norm_cov_params, cov_type=c_type)
print("This summary should give back the same result as __ START: Fixed Effect Model ____'s summary: ")
displayhook(summary.summary())

print(''' ____ END: Verifying Algorithm ____''')



print(''' __ START: VIF (Variance Inflation Factor) ____''')

n_indeps = fixed_effect_mdl.exog.shape[1]
vifs = [ variance_inflation_factor(fixed_effect_mdl.exog, i) for i in range(0, n_indeps) ]
vifs_df = pd.DataFrame(vifs, index=fixed_effect_mdl.exog_names, columns=["VIF"])
vifs_df.index.name = "Independent variables"
displayhook(vifs_df)
''' One recommendation is that if VIF is greater than 5,
    then the independent variable is highly collinear with the other explanatory(=independent) variables, 
    and the parameter(=coefficient=Beta) estimates will have large standard errors because of this.
    In our model, "( PPP - PPP_rem )_pc_{t-1} cent,$" has the VIF over 5. 
    So, we can conclude that "( PPP - PPP_rem )_pc_{t-1} cent,$" is highly correlated with the other independent variables.
'''
barplt = sns.barplot(data=vifs_df.reset_index(), x="VIF", y="Independent variables", hue="Independent variables", palette="icefire", dodge=False)
plt.gca().set_title("VIF for each independent variable")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

print(''' ____ END: VIF (Variance Inflation Factor) ____''')



print(''' __ START: Ridge regression ____''')

n_grid = 50
granularity = 0.003
test_size = [ 0.9, 0.7, 0.5]
sample_num = 100
opt_param_rsqrs = []
opt_betas = []
for ts in test_size:

        rs = ShuffleSplit(n_splits=sample_num, train_size=1-ts, test_size=ts)
        rst = ShuffleSplit(n_splits=sample_num, train_size=1-ts, test_size=ts)

        '''Making traning set and test set'''
        test_one = []
        for train_index, test_index in rs.split(exog.values): test_one.append(test_index)
        test_two = []
        for train_index, test_index in rst.split(exog.values): test_two.append(test_index)
        rs_final = []
        for one, two in zip(test_one, test_two):
                rs_final.append((one, two))

        ridgeCV_alphas = []
        ridgeCV_best_score = []
        ridgeCV_coefs = []
        for g in range(1, n_grid + 1, 1):
                
                fitted_ridge_scores = []
                fitted_ridge_coefs = []
                for test_train in rs_final:
                        fitted_ridge = Ridge(alpha=granularity * g, fit_intercept=False, normalize=False).fit(exog.iloc[test_train[0], :], endg.iloc[test_train[0], :])
                        fitted_ridge_scores.append(fitted_ridge.score(exog.iloc[test_train[1], :], endg.iloc[test_train[1], :]))
                        fitted_ridge_coefs.append(list(fitted_ridge.coef_.flatten()))
                mean_fitted_ridge_scores = np.mean(fitted_ridge_scores)
                fitted_ridge_coefs = list(np.mean(fitted_ridge_coefs, axis=0))

                ridgeCV_alphas.append(granularity * g)
                ridgeCV_best_score.append(mean_fitted_ridge_scores)
                ridgeCV_coefs.append(fitted_ridge_coefs)

        ridge_idx_name = "ridge regression penalties λ, test and train set size: " + str(ts) + ", grid: 0 to " + str(n_grid * granularity) 
        ridge_df_rsqr = pd.DataFrame(ridgeCV_best_score, index=ridgeCV_alphas, columns=["R-squared of the ridge regression"])
        ridge_df_rsqr.index.name = ridge_idx_name

        temp_ridg_df = ridge_df_rsqr[ridge_df_rsqr["R-squared of the ridge regression"].isin([ridge_df_rsqr["R-squared of the ridge regression"].max()])]
        temp_ridg_df["Percentage of the test and training set"] = ts
        temp_ridg_df.index.name = "ridge regression penalties λ, grid: 0 to " + str(n_grid * granularity) 
        opt_param_rsqrs.append(temp_ridg_df)

        ridgeCV_coefs = pd.DataFrame(ridgeCV_coefs, columns=[ var_name + "_Beta" for var_name in reg_var_list[1:] + ps_ctry], index=ridgeCV_alphas)
        ridgeCV_coefs.index.name = ridge_idx_name

        temp_ridg_cof = ridgeCV_coefs.loc[temp_ridg_df.index[0]]
        temp_ridg_cof.index.name = "test and train set size: " + str(ts) + ", grid: 0 to " + str(n_grid * granularity) 
        opt_betas.append(temp_ridg_cof)

        ridgeCV_tstats = []
        for g in range(1, n_grid + 1, 1):
                ridgeCV_res = sm.regression.linear_model.OLSResults(fixed_effect_mdl, \
                                                                    ridgeCV_coefs.loc[g*granularity], \
                                                                    fixed_effect_mdl.normalized_cov_params, \
                                                                    cov_type=c_type)            
                ridgeCV_tstats.append(np.array(ridgeCV_res.tvalues))
        ridgeCV_tstats = pd.DataFrame(ridgeCV_tstats, columns=[ var_name + "_t-Stat" for var_name in reg_var_list[1:] + ps_ctry], index=ridgeCV_alphas)
        ridgeCV_tstats.index.name = ridge_idx_name

        print("Test and train set size: " + str(ts) + ", grid: 0 to " + str(n_grid * granularity))
        displayhook(ridge_df_rsqr)
        print(ridge_df_rsqr.round(decimals=3).to_latex())

        print("-----------------------------------------------------------------------------------------------------------------")
        print("Test and train set size: " + str(ts) + ", grid: 0 to " + str(n_grid * granularity))
        displayhook(ridgeCV_coefs)
        print(ridgeCV_coefs.round(decimals=3).to_latex())

        print("-----------------------------------------------------------------------------------------------------------------")
        print("Test and train set size: " + str(ts) + ", grid: 0 to " + str(n_grid * granularity))                
        displayhook(ridgeCV_tstats)
        print(ridgeCV_tstats.round(decimals=3).to_latex())

        print("-----------------------------------------------------------------------------------------------------------------")
        ''' PLOTS '''

        sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=0.9)

        '''R-squared plot'''

        sns.lineplot(data=ridge_df_rsqr , x=ridge_idx_name, y='R-squared of the ridge regression', color="black")
        # plt.gca().set_title("R-squared in accordance with ridge regression penalties")
        plt.show()

        '''Beta plots'''

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[1]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[1]+"_Beta", color="red")
        # plt.gca().set_title("Coefficient of the remittance in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[2]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[2]+"_Beta", color="blue")
        # plt.gca().set_title("Coefficient of the PPP in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[3]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[3]+"_Beta", color="green")
        # plt.gca().set_title("Coefficient of the unemployment rate in accordance with ridge regression penalties")
        plt.show()

        ridgeCV_ctry_coefs_melted = ridgeCV_coefs[[ var_name + "_Beta" for var_name in ps_ctry]].reset_index().melt(id_vars=[ridge_idx_name], value_vars=[ var_name + "_Beta" for var_name in ps_ctry])
        ridgeCV_ctry_coefs_melted.columns = [ ridge_idx_name, 'Country', 'Ridge estimated Beta (Coefficient)' ]
        sns.lineplot(data=ridgeCV_ctry_coefs_melted, x=ridge_idx_name, y='Ridge estimated Beta (Coefficient)', hue='Country', palette="mako_r")
        # plt.gca().set_title("Beta for every country dummy variable in accordance with ridge regression penalties")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        '''t-Stat plots'''

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[1]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[1]+"_t-Stat", color="red")
        # plt.gca().set_title("t-Stat of the remittance in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[2]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[2]+"_t-Stat", color="blue")
        # plt.gca().set_title("t-Stat of the PPP in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[3]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[3]+"_t-Stat", color="green")
        # plt.gca().set_title("t-Stat of the unemployment rate in accordance with ridge regression penalties")
        plt.show()

        ridgeCV_ctry_tstats_melted = ridgeCV_tstats[[ var_name + "_t-Stat" for var_name in ps_ctry]].reset_index().melt(id_vars=[ridge_idx_name], value_vars=[ var_name + "_t-Stat" for var_name in ps_ctry])
        ridgeCV_ctry_tstats_melted.columns = [ ridge_idx_name, 'Country', 'Ridge t-Stat' ]
        sns.lineplot(data=ridgeCV_ctry_tstats_melted, x=ridge_idx_name, y='Ridge t-Stat', hue='Country', palette="mako_r")
        # plt.gca().set_title("t-Stat for every country dummy variable in accordance with ridge regression penalties")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

opt_param_rsqrs = pd.concat(opt_param_rsqrs, axis=0)
print(opt_param_rsqrs.to_latex())

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    tvals = results.tvalues

    results_df = pd.DataFrame({"p-value":pvals,
                               "Coef":coeff,
                               "t-Stat":tvals
                                })

    # Reordering
    results_df = results_df[["Coef","p-value","t-Stat"]]
    return results_df.round(decimals=3)

dfs_ridge_agg = []
for obta, ts in zip(opt_betas, test_size):
        print("The test size is: "+str(ts))
        ridgeCV_res = sm.regression.linear_model.OLSResults(fixed_effect_mdl, \
                                                            obta, \
                                                            fixed_effect_mdl.normalized_cov_params, \
                                                            cov_type=c_type)
        displayhook(ridgeCV_res.summary())
        dfs_ridge_agg.append(results_summary_to_dataframe(ridgeCV_res))

dfs_ridge_agg = pd.concat(dfs_ridge_agg, axis=1)
print(dfs_ridge_agg.to_latex())

''' PLOTS '''

print(''' ____ END: Ridge regression ____''')



print(''' __ START: Partial Least Squares ____''')

pls_tstats = []
pls_betas = []
r_squared = []
indices = []
num_possible_comp = len(exog.columns)
for n_comp in range(1, num_possible_comp +1, 1):

        pls_mdl = PLSRegression(n_components=n_comp, max_iter=10000, scale=False, tol=1e-10)
        res = pls_mdl.fit(exog, endg)
        print(res.get_params())
        pls_beta = pd.DataFrame(res.coef_, index=fitted_fixed_effect_mdl.params.index, columns=["Beta:PLS"])

        pls_temp = sm.regression.linear_model.OLSResults(fixed_effect_mdl,
                                                         pls_beta["Beta:PLS"],
                                                         normalized_cov_params=fixed_effect_mdl.normalized_cov_params,
                                                         cov_type=c_type)        
        pls_tstats.append(pls_temp.tvalues)
        pls_betas.append(pls_temp.params)
        r_squared.append(pls_mdl.score(exog, endg))
        indices.append(n_comp)

r_squared = pd.DataFrame(np.array([r_squared, indices]), index=["R-squared", "The number of independent scores"]).T
sns.lineplot(data=r_squared, x='The number of independent scores', y='R-squared', color="black")
# plt.gca().set_title("R-squared in accordance with the number of independent scores")
plt.show()

pls_tstats = pd.DataFrame(pls_tstats)
pls_tstats['The number of independent scores'] = indices
pls_tstats = pls_tstats.melt(id_vars=["The number of independent scores"], var_name="indep var", value_name="t-stat")

''' PLOTS '''
pls_tstats_three_fac = pls_tstats[pls_tstats['indep var'].isin(reg_var_list[1:])]
sns.lineplot(data=pls_tstats_three_fac, x='The number of independent scores', y='t-stat', hue='indep var', palette="icefire")
# plt.gca().set_title("T-stat for the remittance, the PPP, and the unemployment rate in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_tstats_dummy_fac = pls_tstats[pls_tstats['indep var'].isin(ps_ctry)]
sns.lineplot(data=pls_tstats_dummy_fac, x='The number of independent scores', y='t-stat', hue='indep var', palette="mako_r")
# plt.gca().set_title("T-stat for every country dummy variable in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_betas = pd.DataFrame(pls_betas)
pls_betas['The number of independent scores'] = indices
pls_betas = pls_betas.melt(id_vars=["The number of independent scores"], var_name="indep var", value_name='PLS Beta (Coefficient)')

pls_betas_three_fac = pls_betas[pls_betas['indep var'].isin(reg_var_list[1:])]
sns.lineplot(data=pls_betas_three_fac, x='The number of independent scores', y='PLS Beta (Coefficient)', hue='indep var', palette="icefire")
# plt.gca().set_title("Beta for the remittance, the PPP, and the unemployment rate in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_betas_dummy_fac = pls_betas[pls_betas['indep var'].isin(ps_ctry)]
sns.lineplot(data=pls_betas_dummy_fac, x='The number of independent scores', y='PLS Beta (Coefficient)', hue='indep var', palette="mako_r")
# plt.gca().set_title("Beta for every country dummy variable in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

''' PLOTS '''

pls_mdl = PLSRegression(n_components=8, max_iter=10000, scale=False, tol=1e-10)
res = pls_mdl.fit(exog, endg)
print(res.get_params())
pls_beta = pd.DataFrame(res.coef_, index=fitted_fixed_effect_mdl.params.index, columns=["Beta:PLS"])

# Comparing the Betas estimated by the Fixed Effect Model and the Partial Least Squares.
betaCompare = pd.concat([pd.DataFrame(fitted_fixed_effect_mdl.params, columns=["Beta:FixedEffectMdl"]), pls_beta], axis=1)
displayhook(betaCompare)
print(betaCompare.round(3).to_latex())

pls_final = sm.regression.linear_model.OLSResults(fixed_effect_mdl,
                                                  pls_beta["Beta:PLS"],
                                                  normalized_cov_params=fixed_effect_mdl.normalized_cov_params,
                                                  cov_type=c_type)
displayhook(pls_final.summary())

df_smry_pls = pd.concat([results_summary_to_dataframe(fitted_fixed_effect_mdl), results_summary_to_dataframe(pls_final)], axis=1)
print(df_smry_pls.to_latex())

print(''' ____ END: Partial Least Squares ____''')

pls_mdl = PLSRegression(n_components=8, max_iter=10000, scale=False, tol=1e-10)
res = pls_mdl.fit(exog, endg)
pinv_pT_W, _ = pinv_extended(np.dot(res.x_loadings_.T, res.x_weights_))
beta = np.dot(np.dot(res.x_weights_, pinv_pT_W), res.y_weights_.T)
print("Coefficient has been reconstructed!: ", res.coef_ - beta)
np.shape(res.x_loadings_)
np.shape(res.x_weights_)

print('''*********************************** END: FIXED EFFECT MODEL, RIDGE REGRESSION, PARTIAL LEAST SQUARES REGRESSION *********************************''')
