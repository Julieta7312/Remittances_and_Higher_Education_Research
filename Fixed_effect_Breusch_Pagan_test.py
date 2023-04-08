''' __ START: Fixed Effect Model ____'''

fixed_effect_mdl = sm.OLS(endg, exog)
fitted_fixed_effect_mdl = fixed_effect_mdl.fit(cov_type=c_type)
displayhook(fitted_fixed_effect_mdl.summary())

''' ____ END: Fixed Effect Model ____'''



''' __ START: Fixed Effect Model: Breusch-Pagan-Test ____'''

fitted_fixed_effect_estimations = ( fitted_fixed_effect_mdl.params[:3] * panel_df[reg_var_list[1:]] ).sum(axis=1)
fitted_fixed_effect_residuals = fitted_fixed_effect_mdl.resid

est_resid = pd.DataFrame(fitted_fixed_effect_estimations, columns=["Fixed effect estimated enrollment rate"]).merge(pd.DataFrame(fitted_fixed_effect_residuals, columns=["residual"]), left_index=True, right_index=True)
est_resid_exog = pd.DataFrame(est_resid).merge(exog, left_index=True, right_index=True)

sns.scatterplot(data=est_resid_exog, x="Fixed effect estimated enrollment rate", y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among Fixed effect model's fitted values and it's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=rem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the remittance and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=ppp_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the PPP and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

sns.scatterplot(data=est_resid_exog, x=unem_reg_name, y='residual', hue='residual', palette="icefire", legend=False)
plt.gca().set_title("Heteroscedasticity among the youth unemployment and the Fixed effect model's residual")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

breusch_pagan_test_results = het_breuschpagan(est_resid_exog['residual'], panel_df[reg_var_list[1:]])
displayhook(pd.DataFrame(breusch_pagan_test_results, columns=["Fixed effect model: Breusch pagan test result"], index=["LM-Stat", "LM p-val", "F-Stat", "F p-val"]))

''' ____ END: Fixed Effect Model: Breusch-Pagan-Test ____'''