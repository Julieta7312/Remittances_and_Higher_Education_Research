''' __ START: Pooled OLS Model ____'''

pooled_ols_mdl = sm.OLS(endg, exog[reg_var_list[1:]])
fitted_pooled_ols_mdl = pooled_ols_mdl.fit(cov_type=c_type)
displayhook(fitted_pooled_ols_mdl.summary())
''' ____ END: Pooled OLS Model ____'''



''' __ START: Pooled OLS Model: Breusch-Pagan-Test ____'''

fitted_pooled_ols_estimations = ( fitted_pooled_ols_mdl.params[:3] * panel_df[reg_var_list[1:]] ).sum(axis=1)
fitted_pooled_ols_residuals = fitted_pooled_ols_mdl.resid

pool_est_resid = pd.DataFrame(fitted_pooled_ols_estimations, columns=["Pooled OLS estimated enrollment rate"]).merge(pd.DataFrame(fitted_pooled_ols_residuals, columns=["residual"]), left_index=True, right_index=True)
pool_est_resid_exog = pd.DataFrame(pool_est_resid).merge(exog, left_index=True, right_index=True)

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)

sns.scatterplot(data=pool_est_resid_exog, x="Pooled OLS estimated enrollment rate", y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among Pooled OLS's fitted values and it's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=rem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the remittance and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=ppp_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the PPP and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=pool_est_resid_exog, x=unem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the youth unemployment and the Pooled OLS's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pool_breusch_pagan_test_results = het_breuschpagan(pool_est_resid_exog['residual'], panel_df[reg_var_list[1:]])
displayhook(pd.DataFrame(pool_breusch_pagan_test_results, columns=["Pooled OLS model: Breusch pagan test result"], index=["LM-Stat", "LM p-val", "F-Stat", "F p-val"]))

''' ____ END: Pooled OLS Model: Breusch-Pagan-Test ____'''