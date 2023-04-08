'''********************************* START: DATA PROCESSING *********************************'''

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

''' Winsorize '''

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

'''*********************************** END: DATA PROCESSING *********************************'''