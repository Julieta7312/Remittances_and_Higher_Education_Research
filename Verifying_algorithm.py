''' __ START: Verifying Algorithm ____'''

print("This proves that the model.wexog is equal to X (=A matrix of independent variables, whitened? exogenous variables): ", np.sum(exog.values) == np.sum(fixed_effect_mdl.wexog))

pseudo_inv_X, singular_vals = pinv_extended(fixed_effect_mdl.wexog) # Return the pseudo inverse of X as well as the singular values used in computation.
norm_cov_params = np.dot(pseudo_inv_X, np.transpose(pseudo_inv_X)) # X^{-1} \dot (X^{-1})^T
pseudo_inv_XTX, singular_values = pinv_extended(np.dot(fixed_effect_mdl.wexog.T, fixed_effect_mdl.wexog))
print( "This shows that fixed_effect_mdl.normalized_cov_params = X^{-1} \dot (X^{-1})^T = ( X^T \dot X )^{-1}: ", np.sum(norm_cov_params) - np.sum(pseudo_inv_XTX) < 1e-13 )

summary = sm.regression.linear_model.OLSResults(fixed_effect_mdl, fitted_fixed_effect_mdl.params, norm_cov_params, cov_type=c_type)
print("This summary should give back the same result as __ START: Fixed Effect Model ____'s summary: ")
displayhook(summary.summary())

''' ____ END: Verifying Algorithm ____'''