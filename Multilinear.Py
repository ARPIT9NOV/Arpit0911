## MUlti linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cars = pd.read_csv("Cars.csv")

cars.head(40)

cars.corr()

## scatterplot diagram along with histogram
import seaborn as sns
sns.pairplot(cars)

cars.columns

##prepairing model cosidering all variables(Observed least square)
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('MPG~HP+VOL+SP+WT', data=cars) .fit()
ml1.params
ml1.summary()


##prepairing model based on Volume

ml2 = smf.ols('MPG~VOL', data=cars) .fit()
ml2.summary() #0.271
#pvalue<0.005 it is significant

#Prepairing model based on Wt
ml3 = smf.ols('MPG~WT', data = cars) .fit()
ml3.summary() #0.268

# Preapairing model based on WT & VOL
ml4 = smf.ols('MPG~VOL+WT', data = cars) .fit()
ml4.summary() 

## Influencing Index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
 
## 76 & 78 is showing highly influenced so remove that line

cars_new = cars.drop(cars.index[[76,78]],axis=0) 

model1 = smf.ols('MPG~WT+VOL+HP+SP', data = cars_new) .fit()
model1.summary()
model1.params

## confidence level 99%
print(model1.conf_int(0.01))

## VIF calculation for variables
rsq_hp = smf.ols('HP~VOL+WT+SP', data = cars_new).fit().rsquared
vif_hp = 1/(1-rsq_hp) #16.33

rsq_vol = smf.ols('VOL~HP+WT+SP', data = cars_new).fit().rsquared
vif_vol = 1/(1-rsq_vol) #564.84

rsq_wt = smf.ols('WT~VOL+HP+SP', data = cars_new).fit().rsquared
vif_wt = 1/(1-rsq_wt) #564.98

rsq_sp = smf.ols('SP~VOL+WT+HP', data = cars_new).fit().rsquared
vif_sp = 1/(1-rsq_hp) #16.33

# Storing vif values in a data frame
d1 = {'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame = pd.DataFrame(d1)
vif_frame

# Added variable plot
sm.graphics.plot_partregress_grid(model1)

# Final model
final_model = smf.ols('MPG~VOL+SP+HP', data = cars_new) .fit()
final_model.params
final_model.summary() # r^2= 0.848

mpg_pred = final_model.predict(cars_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)

## linearity 
# observed values vs fitted values
plt.scatter(cars_new.MPG,mpg_pred, c="r");plt.xlabel("Observed_values");plt.ylabel("Fitted_Values")

# Residuals VS Fitted Values
plt.scatter(mpg_pred,final_model.resid_pearson, c = "r");plt.axhline(y=0,color= "blue");plt.xlabel("Fitted_values");plt.ylabel("Residuals")

########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson)

# QQ plot for residuals 
import pylab
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist='norm', plot=pylab)

############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_model.resid_pearson, c = "r");plt.axhline(y=0,color= "blue");plt.xlabel("Fitted_values");plt.ylabel("Residuals")

### Splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test = train_test_split(cars_new,test_size=0.20)

# preparing the model on train data 
model_train = smf.ols('MPG~VOL+HP+SP',data=cars_train).fit()

#train data prediction
train_pred = model_train.predict(cars_train)

#train residual values
train_resid = train_pred - cars_train.MPG

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values
test_resid = test_pred - cars_test.MPG

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
