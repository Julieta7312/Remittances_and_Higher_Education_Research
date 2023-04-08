''' __ START: Ridge regression ____'''

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
        plt.gca().set_title("R-squared in accordance with ridge regression penalties")
        plt.show()

        '''Beta plots'''

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[1]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[1]+"_Beta", color="red")
        plt.gca().set_title("Coefficient of the remittance in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[2]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[2]+"_Beta", color="blue")
        plt.gca().set_title("Coefficient of the PPP in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_coefs[[reg_var_list[3]+"_Beta"]] , x=ridge_idx_name, y=reg_var_list[3]+"_Beta", color="green")
        plt.gca().set_title("Coefficient of the unemployment rate in accordance with ridge regression penalties")
        plt.show()

        ridgeCV_ctry_coefs_melted = ridgeCV_coefs[[ var_name + "_Beta" for var_name in ps_ctry]].reset_index().melt(id_vars=[ridge_idx_name], value_vars=[ var_name + "_Beta" for var_name in ps_ctry])
        ridgeCV_ctry_coefs_melted.columns = [ ridge_idx_name, 'Country', 'Ridge estimated Beta (Coefficient)' ]
        sns.lineplot(data=ridgeCV_ctry_coefs_melted, x=ridge_idx_name, y='Ridge estimated Beta (Coefficient)', hue='Country', palette="mako_r")
        plt.gca().set_title("Beta for every country dummy variable in accordance with ridge regression penalties")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        '''t-Stat plots'''

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[1]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[1]+"_t-Stat", color="red")
        plt.gca().set_title("t-Stat of the remittance in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[2]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[2]+"_t-Stat", color="blue")
        plt.gca().set_title("t-Stat of the PPP in accordance with ridge regression penalties")
        plt.show()

        sns.lineplot(data=ridgeCV_tstats[[reg_var_list[3]+"_t-Stat"]] , x=ridge_idx_name, y=reg_var_list[3]+"_t-Stat", color="green")
        plt.gca().set_title("t-Stat of the unemployment rate in accordance with ridge regression penalties")
        plt.show()

        ridgeCV_ctry_tstats_melted = ridgeCV_tstats[[ var_name + "_t-Stat" for var_name in ps_ctry]].reset_index().melt(id_vars=[ridge_idx_name], value_vars=[ var_name + "_t-Stat" for var_name in ps_ctry])
        ridgeCV_ctry_tstats_melted.columns = [ ridge_idx_name, 'Country', 'Ridge t-Stat' ]
        sns.lineplot(data=ridgeCV_ctry_tstats_melted, x=ridge_idx_name, y='Ridge t-Stat', hue='Country', palette="mako_r")
        plt.gca().set_title("t-Stat for every country dummy variable in accordance with ridge regression penalties")
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

''' ____ END: Ridge regression ____'''