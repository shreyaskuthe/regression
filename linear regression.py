import pandas as pd
data=pd.read_csv(r'E:\DATA SCIENCE\imarticus\python\datasets\Advertising.csv',index_col=0,header=0)
print(data.head())
#%%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%%
print(data.dtypes)
print(data.shape)
print(data.describe())
#%%%
data.boxplot(column='TV')
#%%%
data.boxplot(column='radio')
#%%%
data.boxplot(column='newspaper')
#%%%
sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',
             kind='reg')
#%%%
#create X and Y
X=data[['TV','radio','newspaper']]
Y=data['sales']
#%%%
sns.distplot(Y,hist=True)
#%%%
"""
#log transformation
import numpy as np
Y_log=np.log(Y)
"""
#sns.distplot(Y_log,hist=True)
#%%%
X.hist(bins=20)
#%%%
from scipy.stats import skew
data_num_skew = X.apply(lambda x: skew(x.dropna()))
data_num_skewed = data_num_skew[(data_num_skew > .75) | (data_num_skew < -.75)]

print(data_num_skew)
print(data_num_skewed)
import numpy as np
# apply log + 1 transformation for all numeric features with skewness over .75
X[data_num_skewed.index] = np.log1p(X[data_num_skewed.index])
#%%%
X.hist(bins=50)
#%%%
import seaborn as sns
corr_df=X.corr(method='pearson')
print(corr_df)
plt.figure(figsize=(5,5))
sns.heatmap(corr_df,vmax=1.0,vmin=-1.0,annot=True)
plt.show()
#%%%
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df["features"] = X.columns
vif_df["VIF Factor"] = [vif(X.values, i) for i in range(X.shape[1])]
vif_df.round(2)
#%%%
from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
#%%%
from sklearn.linear_model import LinearRegression
#create a model object
lm=LinearRegression()
#train the model object
lm.fit(X_train,Y_train)
#print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)
#%%%
#pair the feature names with the coefficients
print(list(zip(X.columns,lm.coef_)))
#%%%
X1=100
X2=100
X3=50
y_pred=3.353291385815151+(0.0437425*X1)+(0.19303708*X2)+(-0.04895137*X3)
print(y_pred)
#%%%
#predict using the model
Y_pred=lm.predict(X_test)
print(Y_pred)
#%%%
new_df=pd.DataFrame()
new_df=X_test
new_df['Actual sales']=Y_test
new_df['Predicted sales']=Y_pred
new_df
#%%%
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
r2=r2_score(Y_test,Y_pred)
print(r2)
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)
adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print(adjusted_r_squared)
#%%%
print(min(Y_test))
print(max(Y_test))
#%%%
new_df=pd.DataFrame()
new_df=X_train
new_df['sales']=Y_train
new_df.shape
#%%%
import statsmodels.formula.api as sm
#create a fitted model with all three features
lm_model=sm.ols(formula='sales~TV+radio+newspaper',data=new_df).fit()
#print the coefficients
print(lm_model.params)
print(lm_model.summary())
#%%%
Y_pred_new=lm_model.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
r2=r2_score(Y_test,Y_pred)
print(r2)
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)
adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print(adjusted_r_squared)
#%%%
new_df1=pd.DataFrame()
new_df1=X_test
new_df1['sales']=Y_test
new_df1.shape
#%%%
import statsmodels.formula.api as sm
#create a fitted model with all three features
lm_model1=sm.ols(formula='sales~TV+radio+newspaper',data=new_df1).fit()
#print the coefficients
print(lm_model1.params)
print(lm_model1.summary())
#%%%
new_df2=pd.DataFrame()
new_df2=X_train
new_df2['sales']=Y_train
new_df2.shape
#%%%
import statsmodels.formula.api as sm
#create a fitted model with all three features
lm_model2=sm.ols(formula='sales~TV+radio',data=new_df2).fit()
#print the coefficients
print(lm_model2.params)
print(lm_model2.summary())
#%%%
new_df3=pd.DataFrame()
new_df3=X_test
new_df3['sales']=Y_test
new_df3.shape
#%%%
import statsmodels.formula.api as sm
#create a fitted model with all three features
lm_model3=sm.ols(formula='sales~TV+radio+newspaper',data=new_df3).fit()
#print the coefficients
print(lm_model3.params)
print(lm_model3.summary())
#%%%
Y_pred_new=lm_model.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
r2=r2_score(Y_test,Y_pred)
print(r2)
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)
adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print(adjusted_r_squared)
#%%%
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

# fitted values (need a constant term for intercept)
model_fitted_y = lm_model.fittedvalues

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'sales', data=new_df, lowess=True)

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
#%%%
import statsmodels.api as stm
import scipy.stats as stats
fig = stm.qqplot(fit=True, line='45')
plt.title('Normal Q-Q')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Standardized Residuals')
plt.show()

#%%%
# normalized residuals
model_norm_residuals = lm_model.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, lowess=True)


plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
#%%%
from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
#%%%
from sklearn.linear_model import Ridge
lm=Ridge()
lm.fit(X_train,Y_train)
#print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)
#%%%
from sklearn.linear_model import Lasso
lm=Lasso()
lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_)
#%%%

