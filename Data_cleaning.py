# Import the packages
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

'''********************************* START: READ & TIDY UP THE DATA *********************************'''

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

'''*********************************** END: READ & TIDY UP THE DATA *********************************'''
