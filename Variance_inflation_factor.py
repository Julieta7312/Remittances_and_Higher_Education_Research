''' __ START: VIF (Variance Inflation Factor) ____'''

n_indeps = fixed_effect_mdl.exog.shape[1]
vifs = [ variance_inflation_factor(fixed_effect_mdl.exog, i) for i in range(0, n_indeps) ]
vifs_df = pd.DataFrame(vifs, index=fixed_effect_mdl.exog_names, columns=["VIF"])
vifs_df.index.name = "Independent variables"
displayhook(vifs_df)
''' One recommendation is that if VIF is greater than 5,
    then the independent variable is highly collinear with other explanatory(=independent) variables, 
    and the parameter(=coefficient=Beta) estimates will have large standard errors because of this.
    In our model, "( PPP - PPP_rem )_pc_{t-1} cent,$" has the VIF over 5. 
    So, we can conclude that "( PPP - PPP_rem )_pc_{t-1} cent,$" is highly correlated with other independent variables.
'''
barplt = sns.barplot(data=vifs_df.reset_index(), x="VIF", y="Independent variables", hue="Independent variables", palette="icefire", dodge=False)
plt.gca().set_title("VIF for each independent variable")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

''' ____ END: VIF (Variance Inflation Factor) ____'''