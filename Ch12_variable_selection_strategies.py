#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import statsmodels.formula.api as smf
sns.set_style('whitegrid')


# In[50]:


path = 'sleuth3csv/case0902.csv'
df = pd.read_csv(path)
df.head()


# ## Partial residuals
# indicates the association of two variables, after getting the effect of a third variable out of the way.
# 
# Let's find the association of log Brain size (lbrain) and log Gestation (lgest) period after removing the effect of log body size (lbody).
# 
# i.e. Partial residual plot of log Brain size vs log Gestation.
# 
# Steps:
# 
# 1) Obtain the estimated coefficients in the linear regression of lbrain on lbody and lgest: $\mu\{lbrain|lbody, lgest\} = \beta_0+\beta_1 lbody + \beta_2 lgest$.
# 
# 2) Compute the partial residuals as $pres = lbrain-\hat{\beta}_0-\hat{\beta}_1 lbody$.
# * or $pres = res+\hat{\beta}_2 lgest$
# 
# 3) Plot the partial residuals vs lgest

# In[52]:


model = smf.ols("np.log(Brain) ~ np.log(Body)+np.log(Gestation)", data = df)
results = model.fit()
print(results.summary())


# In[53]:


fig, axes = plt.subplots(ncols = 2, figsize = (20,5), sharey=True)
sns.scatterplot(x = np.log(df["Gestation"]), y = np.log(df["Brain"]), ax = axes[0])
sns.scatterplot(x = np.log(df["Gestation"]) , y = results.resid+0.6678*np.log(df["Gestation"]), ax = axes[1]).set(title='Partial residual plot')
plt.show()


# After getting the effect of log body weight out of the way, less of an association remains between lbrain and lgest.
# 
# But some association does persist and a linear term should be adequate to model it.

# ## State Average SAT Scores

# In[10]:


path = 'sleuth3csv/case1201.csv'
df = pd.read_csv(path)
display(df)


# Data is from the year 1982.
# 
# ***Takers*** - percentage of the total eligible students (high school seniors) in the state who took the exam.
# 
# ***Income*** - median income of families of test-takers, in hundreds of dollars.
# 
# ***Years*** - average number of years that the test-takers had formal studies in social sciences, natural sciences, and humanities.
# 
# ***Public*** - percentage of the test-takers who attended public secondary schools.
# 
# ***Expend*** - total state expenditure on secondary schools, expressed in hundreds of dollars per student.
# 
# ***Rank*** - median percentile ranking of the test-takers within their secondary school classes.

# In[8]:


## out of a possible 1600
print(df.query('SAT == SAT.min()'))
print(df.query('SAT == SAT.max()'))


# Note that the states with high average (mostly midwestern states) SATs had low percentages of takers.
# 
# Explanation: Only their best students planning to attend college out of state take the SAT exams.

# ## Research questions:
# 
# 1) After accounting for the percentage of takers and the median class rank of the takers, 
# 
# * which variables are associated with state SAT scores?
# 
# * how do the states rank?
# 
# 2) Which states perform best for the amount of money they spend?

# ## Sex Discrimination in Employment
# Data on employees from one job category of a bank that was sued for sex discrimination. These are the same 32 male and 61 female employees, hired between 1965 and 1975.

# In[11]:


path = 'sleuth3csv/case1202.csv'
df = pd.read_csv(path)
df.head()


# ***Bas*** - Basic salary at the time of hire
# 
# ***Sal77*** - Salary as of March 1977
# 
# ***Senior*** - Seniority (months since first hired)
# 
# ***Age*** - Age (months)
# 
# ***Educ*** - Education (year)
# 
# ***Exper*** -  Experience prior to employment with the bank (months)

# ## Research questions:
# 
# 1) Did the females receive lower starting salaries than similarly qualified and similarly experienced males?
# 
# 2) After accounting for measures of performance, did females receive smaller pay increases than males?

# ## Purposes of regression
# 
# 1) Modeling a large set of explanatory variables: determine a subset of variables that play a role in the regression model.
# 
# 2) Explain the data: no well defined question is posed, but something like, “which variables are important to explain ‘___’  accounting for these ‘____’ variables.”   
# * This can be exceptionally difficult at changing one input variable can change your results, and there are many different models.
# * Coefficient interpretation, "measure of the effect of one variable while holding all other variables fixed." But, the increase in one variable almost always changes others.
# 
# 3) Prediction: The goal is to form the most accurate outputs based on a subset of input variables.
# * No interpretation of a set of explanatory variables chosen or their coefficients is needed.

# ## Was the bank guilt of gender discrimination?
# 
# 1) Idea: 
# * Use variable selection methods to select ‘the best’ subset of the variables for the output base salary.
# * Once this subset/model is selected we will add gender to the model.
# * Interpretation of gender coefficient: association between gender and salary after accounting for the effects of the other explanatory variables.
# * If this gender variable is statistically significant it means that base salary cannot be explained by other variables and the bank is guilty of gender discrimination.
# 
# 2) The variable selection methods are useful for this purpose. 
# * While a specific model only includes only a subset of all the variables. The final model has been adjusted for other variables as they were given a chance to be in the model (prior to inclusion of gender).
# 

# ## Loss of Precision
# Precision in estimating coefficients and prediction may decrease if too many explanatory variables are included in the model. 
# 
# 
# 
# $$SE(\hat{\beta_X}_\text{ Multiple Linear Model}) = SE(\hat{\beta_X}_\text{ Simple Linear Model})\sqrt{\text{ Mean square ratio}*\text{ Variance inflation factor}}$$
# 
# $\text{Mean square ratio} = \frac{\hat{\sigma}^2_\text{Multiple Linear Model}}{\hat{\sigma}^2_\text{Simple Linear Model}}$
# 
# $\text{Mean square ratio}<1$ if Multiple Linear Model has useful predictors.
# 
# $$\text{Variance Inflation Factor} = VIF = \frac{1}{1-R^2_X}$$
# 
# $R^2_X$ = the proportion of the variation in X explained by its linear relationship to other explanatory variables in the model. 
# 
# When X can be explained well by the additional variables (multicollinearity) the VIF can be very high.
# 
# 
# If VIF < 1 then the additional variable is not related to the the other input variables.
# 
# If VIF > 1 then the additional variable is are related to the the other input variables.
# 
# A general guideline is that a VIF larger than 5 or 10 is large, indicating that the model has problems estimating the coefficient.

# ## A Strategy for Dealing with Many Explanatory Variables
# 
# 1) Identify the key objectives.
# 
# 2) Screen the available variables, deciding on a list that is sensitive to the objectives and excludes obvious redundancies.
# 
# 3) Perform exploratory analysis, examining graphical displays and correlation coefficients.
# 
# 4) Perform transformations as necessary.
# 
# 5) Examine a residual plot after fitting a rich model, performing further transformations and considering outliers.
# 
# 6) Use a computer-assisted technique for finding a suitable subset of explanatory variables.
# 
# 7) Proceed with the analysis, using the selected explanatory variables.

# In[54]:


path = 'sleuth3csv/case1201.csv'
df = pd.read_csv(path)
df.head()


# In[19]:


sns.pairplot(df[["Takers","Rank","Years","Income","Public","Expend","SAT"]], corner=True)
plt.show()


# * Indicates a nonlinear relationship between SAT and percentage of takers.
# 
# * Some potential outliers in the variables public school percentage and state expenditure.

# In[20]:


plt.figure(figsize = (7,4))
sns.scatterplot(x="Takers", y="SAT", data = df).set( xscale="log")
plt.show()


# In[69]:


display(df[df.Public<60])
display(df[df.Expend>40])

filter_ = ((df.Public<60)|(df.Expend>40))


# These require further examination after a model has been fit.
# 
# We test the model:
# 
# $$\mu\{SAT|lakers,rank\} = \beta_0+\beta_1 ltakers+\beta_2 rank$$
# 
# excluding Louisiana and Alaska.

# In[71]:


model = smf.ols("SAT ~ np.log(Takers)+Rank", data = df[~filter_])
results = model.fit()
print(results.summary())


# These two variables explain 81.1% of the variation in SAT averages.
# 
# We use a partial residual plot to examine the effect of Expenditure after Taker and Rank are accounted for. 

# In[117]:


model1 = smf.ols("SAT ~ np.log(Takers)+Rank+Expend", data = df)
results1 = model1.fit()
print(results1.summary())


# In[118]:


pres1 = results1.resid+2.8464*np.log(df['Expend'])
pres2 = pres1.drop([28]) #dropping Alaska


# In[119]:


expd = pd.concat([df['Expend'],df.drop([28])['Expend']])

pres = pd.concat([pres1,pres2])

temp_df = pd.DataFrame()

temp_df["expd"] = expd
temp_df["pres"] = pres

temp_df["class"] = "exclude_Alaska"

temp_df.reset_index(inplace=True)

temp_df.loc[0:49,"class"] = "include_Alaska"

temp_df.head()


# In[123]:


sns.lmplot(data=temp_df, x="expd", y="pres", hue="class", ci=None,  
           facet_kws=dict(sharex=False, sharey=False), height=4)
plt.show()


# There is noticeable effect of expenditure, after the two adjustment variables are accounted for (slope of orange line).
# 
# It demonstrates that Alaska is very influential in any fit involving expenditure.
# 
# Now we calculate the leverage and studentized residual for the observation Alaska

# In[121]:


#create instance of influence
influence = results1.get_influence()

#leverage (hat values)
leverage = influence.hat_matrix_diag[28]
student_resid = influence.resid_studentized_internal[28]

print(leverage)
print(student_resid)


# Case influence statistics confirm that Alaska is influential.
# 
# Costs of heating school buildings and transporting teachers and students long distances are far greater in Alaska.
# 
# These expenses do not contribute directly to educational quality.
# 
# So we set Alaska aside for the remainder of the analysis. 

# ## Sequential variable selection
# 
# 1) Forward selection 
# 
# * Starts with a constant mean as the starting model, and adds explanatory variables one at a time until no further variables improve the fit. 
# 
# * The criteria for a new variable to enter the model is the largest F-statistic greater than 4.
# 
# 2) Backward Elimination 
# * Starts with all variables as the starting model. 
# * The criteria to remove a variable is to take the smallest F-statistic to remove that is 4 or less. 
# * We remove variables until no variable can be removed, arriving at the constant mean model.

# Each categorical factor is represented in a regression model by a set of indicator variables. 
# 
# In the absence of a good reason for doing otherwise, this set should be added or removed as a single unit.

# In[18]:


path = 'sleuth3csv/case1201.csv'
df = read.csv(path)
head(df)


# In[4]:


install.packages('leaps')


# In[19]:


df = df[-c(29), ]


# In[27]:


library(leaps)

#Forward model selection
# nvmax - maximum number of predictors to incorporate in the model.
regfit.fwd=regsubsets(df$SAT~df$Income+log(df$Takers)+df$Years+df$Public+df$Expend+df$Rank,data = df, nvmax=6, method ="forward")
#backward or forward


# In[28]:


summary(regfit.fwd)


# In the output “” means the variable is omitted and “*” means the variable is included
#  
# From the first row the largest F-to-enter is log(Takers), so it is taken as the next current model.
# 
# Next, all models that include T and one other variable are examined.
# 
# Expend gives the next largest F-value with the current model.
# 
# and so on.
# 
# But how many variables should I use?

# In[36]:


model1 = lm(df$SAT~log(df$Takers)+df$Expend,data = df)
summary(model1)


# In[37]:


model2 = lm(df$SAT~log(df$Takers)+df$Expend+df$Years,data = df)
summary(model2)


# In model1, both log(Takers) and Expend variables are significant.
# 
# But in model2, Years variable is not significant.
# 
# Thus based on the forward selection method, at this step, model1 is our choice.

# In[39]:


library(leaps)

#Backward model selection
# nvmax - maximum number of predictors to incorporate in the model.
regfit.bwd=regsubsets(df$SAT~df$Income+log(df$Takers)+df$Years+df$Public+df$Expend+df$Rank,data = df, nvmax=6, method ="backward")
#backward or forward


# In[40]:


summary(regfit.bwd)


# In[38]:


#5 variable model
model1 = lm(df$SAT~df$Income+log(df$Takers)+df$Years+df$Expend+df$Rank,data = df)
summary(model1)


# In[42]:


#4 variable model (income variable is excluded)
model2 = lm(df$SAT~log(df$Takers)+df$Years+df$Expend+df$Rank,data = df)
summary(model2)


# In model1, Income variable is not significant so model2 is our choice.

# ## Model selection among all subsets
# 1) Adjusted R^2
# 
# 2) Cp Statistic and Cp plot
# 
# 3) Akaike and Bayesian Information Criteria (AIC and BIC)

# Cp statistic
# 
# Cp focuses on trade-off between bias due to excluding variables and extra variance due to including too many.
# 
# $$Bias\{\hat{Y}_i\} = \mu\{\hat{Y}_i\}-\mu\{Y_i\}$$
# 
# $$MSE\{\hat{Y}_i\}=(Bias\{\hat{Y}_i\})^2+Var\{\hat{Y}_i\}$$
# 
# $$TotalMSE=\sum_{i=1}^nMSE\{\hat{Y}_i\}$$
# 
# $$Cp_{\text{stat}} = \frac{TotalMSE}{\sigma^2}$$
# 
# 
# $Cp_{\text{stat}}$ is computed as 
# 
# $$Cp_{\text{stat}} = p+(n-p)\frac{(\hat{\sigma}^2-\hat{\sigma}^2_{\text{full}})}{\hat{\sigma}^2_{\text{full}}}$$
# 
# where $\hat{\sigma}$ is the estimate of $\sigma$ from the tentative model,
# 
# $\hat{\sigma}_{\text{full}}$ is the estimate of $\sigma$ from the model with all possible variables.
# 
# Models with small Cp statistics are looked on more favorably.
# 
# A model without bias should have Cp approximately equal to p.

# Akaike and Bayesian Information Criteria
# 
# | Criterion to make small | SS Res. part + | Penalty for number of terms |
# | --- | --- | --- |
# | $$Cp =$$ | $$\frac{\text{SSRes}}{\hat{\sigma}^2_{\text{full}}}-n+$$ | $$2p$$ |
# | $$BIC =$$ | $$n\log\big(\frac{\text{SSRes}}{n}\big)+$$ | $$\log(n)(p+1)$$ |
# | $$AIC =$$ | $$n\log\big(\frac{\text{SSRes}}{n}\big)+$$ | $$2(p+1)$$ |
# 
# $BIC>AIC$ for all sample sizes larger than 7.
# 
# $BIC$ tends to favor models with fewer variables than $AIC$.
# 
# $AIC$ seems more appropriate if there are **not** too many unnecessary variables.
# 
# $BIC$ seems more appropriate if there are many unnecessary variables. 
# 
# No criterion can be superior in all situations.

# In[2]:


import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[17]:


path = 'sleuth3csv/case1201.csv'
SATdf = pd.read_csv(path)
SATdf = SATdf.drop([28])
SATdf.shape


# In[18]:


SATdf.drop(columns = "State", inplace = True)
SATdf["Takers"] = np.log(SATdf["Takers"])


# [Link for the full notebook ](https://xavierbourretsicotte.github.io/subset_selection.html)

# In[6]:


def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared


# In[46]:


#Implementing Best subset selection (using itertools.combinations)

#Importing tqdm for the progress bar
from tqdm import tnrange, tqdm_notebook

#Initialization variables
Y = SATdf.SAT
X = SATdf.drop(columns = 'SAT', axis = 1)
k = 7
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#Looping over k = 1 to k = 11 features in X
for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):

    #Looping over all possible combinations: from 11 choose k
    for combo in itertools.combinations(X.columns,k):
        tmp_result = fit_linear_reg(X[list(combo)],Y)   #Store temp result 
        RSS_list.append(tmp_result[0])                  #Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   

#Store in DataFrame
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})


# In[47]:


#Finding the best subsets for each number of features
#Using the smallest RSS value, or the largest R_squared value
df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
display(df_min.head(4))
display(df_max.head(4))


# In[48]:


df


# In[49]:


n = SATdf.shape[0]

df['R_squared_adj'] = 1 - ( (1 - df['R_squared'])*(n-1)/(n-df['numb_features'] -1))
df["BIC"] = n*np.log(df.RSS/n)+np.log(n)*(df.numb_features+1)
df["AIC"] = n*np.log(df.RSS/n)+2*(df.numb_features+1)
df.sort_values("BIC")


# In[ ]:




