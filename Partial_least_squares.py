''' __ START: Partial Least Squares ____'''

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
plt.gca().set_title("R-squared in accordance with the number of independent scores")
plt.show()

pls_tstats = pd.DataFrame(pls_tstats)
pls_tstats['The number of independent scores'] = indices
pls_tstats = pls_tstats.melt(id_vars=["The number of independent scores"], var_name="indep var", value_name="t-stat")

''' PLOTS '''
pls_tstats_three_fac = pls_tstats[pls_tstats['indep var'].isin(reg_var_list[1:])]
sns.lineplot(data=pls_tstats_three_fac, x='The number of independent scores', y='t-stat', hue='indep var', palette="icefire")
plt.gca().set_title("T-stat for the remittance, the PPP, and the unemployment rate in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_tstats_dummy_fac = pls_tstats[pls_tstats['indep var'].isin(ps_ctry)]
sns.lineplot(data=pls_tstats_dummy_fac, x='The number of independent scores', y='t-stat', hue='indep var', palette="mako_r")
plt.gca().set_title("T-stat for every country dummy variable in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_betas = pd.DataFrame(pls_betas)
pls_betas['The number of independent scores'] = indices
pls_betas = pls_betas.melt(id_vars=["The number of independent scores"], var_name="indep var", value_name='PLS Beta (Coefficient)')

pls_betas_three_fac = pls_betas[pls_betas['indep var'].isin(reg_var_list[1:])]
sns.lineplot(data=pls_betas_three_fac, x='The number of independent scores', y='PLS Beta (Coefficient)', hue='indep var', palette="icefire")
plt.gca().set_title("Beta for the remittance, the PPP, and the unemployment rate in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

pls_betas_dummy_fac = pls_betas[pls_betas['indep var'].isin(ps_ctry)]
sns.lineplot(data=pls_betas_dummy_fac, x='The number of independent scores', y='PLS Beta (Coefficient)', hue='indep var', palette="mako_r")
plt.gca().set_title("Beta for every country dummy variable in accordance with the number of independent scores")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

''' PLOTS '''

pls_mdl = PLSRegression(n_components=8, max_iter=10000, scale=False, tol=1e-10)
res = pls_mdl.fit(exog, endg)
print(res.get_params())
pls_beta = pd.DataFrame(res.coef_, index=fitted_fixed_effect_mdl.params.index, columns=["Beta:PLS"])

# Comparing the Betas estimated by the Fixed Effect Model and the Partial Least Squares
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

''' ____ END: Partial Least Squares ____'''

pls_mdl = PLSRegression(n_components=8, max_iter=10000, scale=False, tol=1e-10)
res = pls_mdl.fit(exog, endg)
pinv_pT_W, _ = pinv_extended(np.dot(res.x_loadings_.T, res.x_weights_))
beta = np.dot(np.dot(res.x_weights_, pinv_pT_W), res.y_weights_.T)
print("Coefficient has been reconstructed!: ", res.coef_ - beta)
np.shape(res.x_loadings_)
np.shape(res.x_weights_)