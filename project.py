import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
from warnings import filterwarnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
filterwarnings('ignore')
train_data= pd.read_excel("Data_Train.xlsx")

train_data.dropna(inplace=True)
df=train_data.isna().sum()

df=train_data.dtypes
def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])

#print(train_data.columns)

for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)
#print(train_data.columns)

train_data['Journey_day']=train_data['Date_of_Journey'].dt.day
train_data['Journey_month']=train_data['Date_of_Journey'].dt.month
train_data.drop('Date_of_Journey', axis=1, inplace=True)

def extract_hour(df,col):
    df[col+"_hour"]=df[col].dt.hour
    
def extract_min(df,col):
    df[col+"_minute"]=df[col].dt.minute
def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)
    
extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_column(train_data,'Dep_Time')

extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
drop_column(train_data,'Arrival_Time')

duration=list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:                   
            duration[i]=duration[i] + ' 0m'      
        else:
            duration[i]='0h '+ duration[i]      
train_data['Duration']=duration

def hour(x):
    return x.split(' ')[0][0:-1]
def Min(x):
    return x.split(' ')[1][0:-1]
train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(Min)
drop_column(train_data,'Duration')

train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)


df = train_data.dtypes
cat_col=[col for col in train_data.columns if train_data[col].dtype=='object']
cont_col=[col for col in train_data.columns if train_data[col].dtype!='O']





categorical=train_data[cat_col]
#categorical['Airline'].value_counts()

plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Airline',data=train_data.sort_values('Price',ascending=False)) 
plt.figure(figsize=(15,5))
Airline=pd.get_dummies(categorical['Airline'],drop_first=True)
sns.boxplot(y='Price',x='Total_Stops',data=train_data.sort_values('Price',ascending=False))
Source=pd.get_dummies(categorical['Source'], drop_first=True)



Destination=pd.get_dummies(categorical['Destination'], drop_first=True)


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]

categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)


encoder=LabelEncoder()

for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])
    
drop_column(categorical,'Route')
drop_column(categorical,'Additional_Info')

dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)

data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)

drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')

pd.set_option('display.max_columns',35)

def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
plt.figure(figsize=(30,20))
plot(data_train,'Price')


data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])


X=data_train.drop('Price',axis=1)
y=data_train['Price']

# =============================================================================
# from sklearn.feature_selection import mutual_info_classif
# imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
# 
# imp.columns=['importance']
# imp.sort_values(by='importance',ascending=False)
# =============================================================================

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn import metrics
import pickle
def predict(ml_model,dump):
    model=ml_model.fit(X_train,y_train)
    print("Training score: {}".format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('predictions are: \n {}'.format(y_prediction))
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score of {}: {}'.format(ml_model,r2_score))
# =============================================================================
#     print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
#     print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
#     print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
# =============================================================================
    sns.distplot(y_test-y_prediction)
    
    if dump==1:
        file=open("C:\\Users\\hp\\Desktop\\DS\\airline\\model.pkl",'wb')
        pickle.dump(model,file)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

predict(RandomForestRegressor(),1)
predict(LinearRegression(),0)
predict(KNeighborsRegressor(),0)

predict(DecisionTreeRegressor(),0)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]

# Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]

random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
'max_depth':max_depth,
    'min_samples_split':min_samples_split
}

reg_rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)

rf_random.fit(X_train,y_train)

rf_random.best_params_
prediction=rf_random.predict(X_test)
sns.distplot(y_test-prediction)