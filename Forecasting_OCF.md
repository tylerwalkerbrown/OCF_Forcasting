```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import explained_variance_score,r2_score
from time import time
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, date
import calendar
from sklearn.model_selection import GridSearchCV
  
import warnings
warnings.filterwarnings('ignore')
```


```python
import mysql.connector
import sqlite3
# import the module
import pymysql
from sqlalchemy import create_engine
```


```python
# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{password}@localhost/{database}"
                       .format(user = 'root',
                              password = '',
                              database = ''))
```


```python
#Connector information
mydb = mysql.connector.connect(host = "Tylers-MacBook-Pro.local",
              user = 'root',
              password = '',
              database = ''
              )
```


```python
query = """SELECT date,product_cat,sum(total) as day_total,sum(quantity) as quantity_sold 
FROM OCF.master_table 
where date > '2021-01-01' and date < '2022-01-01'
group by product_cat,date"""
```


```python
df = pd.read_sql(query,con=engine)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>product_cat</th>
      <th>day_total</th>
      <th>quantity_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>winter rye</td>
      <td>854.60</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-04</td>
      <td>tube_tap</td>
      <td>597.55</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-04</td>
      <td>greensand</td>
      <td>118.10</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-05</td>
      <td>winter rye</td>
      <td>203.02</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>tube_tap</td>
      <td>209.95</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 452 entries, 0 to 451
    Data columns (total 4 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   date           452 non-null    datetime64[ns]
     1   product_cat    452 non-null    object        
     2   day_total      452 non-null    float64       
     3   quantity_sold  452 non-null    float64       
    dtypes: datetime64[ns](1), float64(2), object(1)
    memory usage: 14.2+ KB



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 452 entries, 0 to 451
    Data columns (total 4 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   date           452 non-null    datetime64[ns]
     1   product_cat    452 non-null    object        
     2   day_total      452 non-null    float64       
     3   quantity_sold  452 non-null    float64       
    dtypes: datetime64[ns](1), float64(2), object(1)
    memory usage: 14.2+ KB



```python
#Splitting data in date/time to get the independent characteristics of the date string
parts = df["date"].astype('str').str.split("-", n = 3, expand = True)
df["year"]= parts[0].astype('int')
df["month"]= parts[1].astype('int')
df["day"]= parts[2].astype('int')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>product_cat</th>
      <th>day_total</th>
      <th>quantity_sold</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>winter rye</td>
      <td>854.60</td>
      <td>50.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-04</td>
      <td>tube_tap</td>
      <td>597.55</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-04</td>
      <td>greensand</td>
      <td>118.10</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-05</td>
      <td>winter rye</td>
      <td>203.02</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>tube_tap</td>
      <td>209.95</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
from datetime import datetime
import calendar
      
def weekend_or_weekday(year,month,day):
      
    d = datetime(year,month,day)
    if d.weekday()>4:
        return 1
    else:
        return 0
df['weekend'] = df.apply(lambda x:weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>product_cat</th>
      <th>day_total</th>
      <th>quantity_sold</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>winter rye</td>
      <td>854.60</td>
      <td>50.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-04</td>
      <td>tube_tap</td>
      <td>597.55</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-04</td>
      <td>greensand</td>
      <td>118.10</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-05</td>
      <td>winter rye</td>
      <td>203.02</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>tube_tap</td>
      <td>209.95</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Cyclical feature
df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>product_cat</th>
      <th>day_total</th>
      <th>quantity_sold</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekend</th>
      <th>m1</th>
      <th>m2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>winter rye</td>
      <td>854.60</td>
      <td>50.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-04</td>
      <td>tube_tap</td>
      <td>597.55</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-04</td>
      <td>greensand</td>
      <td>118.10</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-05</td>
      <td>winter rye</td>
      <td>203.02</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>tube_tap</td>
      <td>209.95</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>447</th>
      <td>2021-04-25</td>
      <td>winter rye</td>
      <td>8.73</td>
      <td>1.0</td>
      <td>2021</td>
      <td>4</td>
      <td>25</td>
      <td>1</td>
      <td>0.866025</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>448</th>
      <td>2021-04-25</td>
      <td>flowers</td>
      <td>30.60</td>
      <td>1.0</td>
      <td>2021</td>
      <td>4</td>
      <td>25</td>
      <td>1</td>
      <td>0.866025</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>449</th>
      <td>2021-04-25</td>
      <td>20-20-20</td>
      <td>27.18</td>
      <td>2.0</td>
      <td>2021</td>
      <td>4</td>
      <td>25</td>
      <td>1</td>
      <td>0.866025</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>450</th>
      <td>2021-04-25</td>
      <td>greensand</td>
      <td>143.62</td>
      <td>3.0</td>
      <td>2021</td>
      <td>4</td>
      <td>25</td>
      <td>1</td>
      <td>0.866025</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>451</th>
      <td>2021-04-25</td>
      <td>corn gluten</td>
      <td>66.28</td>
      <td>2.0</td>
      <td>2021</td>
      <td>4</td>
      <td>25</td>
      <td>1</td>
      <td>0.866025</td>
      <td>-0.500000</td>
    </tr>
  </tbody>
</table>
<p>452 rows × 10 columns</p>
</div>




```python
#Indicates which day of the week we are in 
def which_day(year, month, day):
      
    d = datetime(year,month,day)
    return d.weekday()
  
df['weekday'] = df.apply(lambda x: which_day(x['year'],
                                                      x['month'],
                                                      x['day']),
                                   axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>product_cat</th>
      <th>day_total</th>
      <th>quantity_sold</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekend</th>
      <th>m1</th>
      <th>m2</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>winter rye</td>
      <td>854.60</td>
      <td>50.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-04</td>
      <td>tube_tap</td>
      <td>597.55</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-04</td>
      <td>greensand</td>
      <td>118.10</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-05</td>
      <td>winter rye</td>
      <td>203.02</td>
      <td>13.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>tube_tap</td>
      <td>209.95</td>
      <td>5.0</td>
      <td>2021</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

Creating aggregated df to look at total sales for the day rather than by  product category and day.


```python
df_agg = df.groupby(['date']).sum()
df_agg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_total</th>
      <th>quantity_sold</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekend</th>
      <th>m1</th>
      <th>m2</th>
      <th>weekday</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-04</th>
      <td>1570.25</td>
      <td>68.0</td>
      <td>6063</td>
      <td>3</td>
      <td>12</td>
      <td>0</td>
      <td>1.5</td>
      <td>2.598076</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>480.94</td>
      <td>20.0</td>
      <td>8084</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>2.0</td>
      <td>3.464102</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>50.27</td>
      <td>3.0</td>
      <td>4042</td>
      <td>2</td>
      <td>12</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.732051</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>15.29</td>
      <td>1.0</td>
      <td>2021</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2021-01-08</th>
      <td>292.22</td>
      <td>13.0</td>
      <td>6063</td>
      <td>3</td>
      <td>24</td>
      <td>0</td>
      <td>1.5</td>
      <td>2.598076</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



Here we look at the outliers that we have in the data set. You can notice a lot of upper bound outliers in the total dataset. We will have to remove the outliers to see where the densities lie. 


```python
plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.distplot(df_agg['day_total'])

plt.subplot(1, 2, 2)
sb.boxplot(df_agg['day_total'])
plt.show()
```


    
![png](output_17_0.png)
    



```python
plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.distplot(df['day_total'])

plt.subplot(1, 2, 2)
sb.boxplot(df['day_total'])
plt.show()
```


    
![png](output_18_0.png)
    


# Benchmarking 


```python
from sklearn.model_selection import train_test_split
```


```python
#Creating a model df that we will use 
model_df = df[df['year'] == 2021][['product_cat','day_total','quantity_sold','month','day','weekend','m1','m2','weekday']]
```


```python
#Encoding dummies for products 
model_df = pd.get_dummies(model_df, columns = ['product_cat'])
```


```python
model_df = model_df[['quantity_sold', 'month', 'day', 'weekend', 'm1', 'm2',
       'weekday', 'product_cat_20-20-20', 'product_cat_alfalfa',
       'product_cat_aluminium sulfate', 'product_cat_buckwheat',
       'product_cat_corn gluten', 'product_cat_flowers',
       'product_cat_garden staples', 'product_cat_greensand',
       'product_cat_oyster shell', 'product_cat_tube_tap',
       'product_cat_urea', 'product_cat_winter rye']]
```


```python
x_train,x_test,y_train,y_test = train_test_split(model_df.loc[:, model_df.columns != 'quantity_sold'],
                                                 model_df['quantity_sold'], test_size = 0.25, random_state = 0)
```


```python
# Standardization the features for stable and fast training.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

# Model Testing


```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
```


```python
#creating list of regresors to loop through and test 
regressors = [
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    LinearRegression(),
    XGBRegressor(),
    Lasso(),
    Ridge()
]
```


```python
head = 10
for model in regressors[:head]:
    start = time()
    model.fit(x_train, y_train)
    train_time = time() - start
    start = time()
    y_pred = model.predict(x_test)
    predict_time = time()-start    
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, y_pred))
    print("\tMean absolute percentage error:", mape(y_test, y_pred))
    print("\tR2 score:", r2_score(y_test, y_pred))
    print()
```

    GradientBoostingRegressor()
    	Training time: 0.036s
    	Prediction time: 0.000s
    	Explained variance: 0.2516397360657777
    	Mean absolute percentage error: 1.3481535507062208
    	R2 score: 0.24531099211676521
    
    ExtraTreesRegressor()
    	Training time: 0.081s
    	Prediction time: 0.006s
    	Explained variance: 0.06469783546617891
    	Mean absolute percentage error: 1.3641067571736494
    	R2 score: 0.0623682379412176
    
    RandomForestRegressor()
    	Training time: 0.093s
    	Prediction time: 0.004s
    	Explained variance: 0.22085317283082528
    	Mean absolute percentage error: 1.5044985894578553
    	R2 score: 0.2034464258191533
    
    DecisionTreeRegressor()
    	Training time: 0.001s
    	Prediction time: 0.000s
    	Explained variance: -0.4613865841653786
    	Mean absolute percentage error: 1.4902592187730137
    	R2 score: -0.46958796231191147
    
    LinearRegression()
    	Training time: 0.004s
    	Prediction time: 0.000s
    	Explained variance: 0.13683067131076687
    	Mean absolute percentage error: 1.6001986070745564
    	R2 score: 0.13199646377370522
    
    XGBRegressor(base_score=None, booster=None, callbacks=None,
                 colsample_bylevel=None, colsample_bynode=None,
                 colsample_bytree=None, early_stopping_rounds=None,
                 enable_categorical=False, eval_metric=None, feature_types=None,
                 gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
                 interaction_constraints=None, learning_rate=None, max_bin=None,
                 max_cat_threshold=None, max_cat_to_onehot=None,
                 max_delta_step=None, max_depth=None, max_leaves=None,
                 min_child_weight=None, missing=nan, monotone_constraints=None,
                 n_estimators=100, n_jobs=None, num_parallel_tree=None,
                 predictor=None, random_state=None, ...)
    	Training time: 0.513s
    	Prediction time: 0.001s
    	Explained variance: -0.1964501594384298
    	Mean absolute percentage error: 1.5403856041201127
    	R2 score: -0.2049859657694768
    
    Lasso()
    	Training time: 0.001s
    	Prediction time: 0.000s
    	Explained variance: 0.009485306885784528
    	Mean absolute percentage error: 1.9356468304787937
    	R2 score: -0.005636506090897386
    
    Ridge()
    	Training time: 0.000s
    	Prediction time: 0.000s
    	Explained variance: 0.14662290705996417
    	Mean absolute percentage error: 1.5891488990977964
    	R2 score: 0.14235482021610568
    



```python
all_models = {}

for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(learning_rate = 0.05, loss = 'ls', max_features = 'sqrt', alpha=alpha)
    all_models["q %1.2f" % alpha] = gbr.fit(x_train, y_train)

all_models["mse"] = gbr.fit(x_train, y_train)
```


```python
y_pred = all_models["mse"].predict(x_test)
y_lower = all_models["q 0.05"].predict(x_test)
y_upper = all_models["q 0.95"].predict(x_test)
y_med = all_models["q 0.50"].predict(x_test)
```


```python
mape(y_test, y_lower)

```




    1.4524761230431829



# Time to Predict!


```python
test = """SELECT ROW_NUMBER() OVER (ORDER BY date) as row_num,product_cat,date
FROM OCF.master_table 
where date > '2022-01-01'
group by product_cat,date"""
```


```python
#Test frame to predict
test = pd.read_sql(test, con = engine)
```


```python
test_og = """SELECT ROW_NUMBER() OVER (ORDER BY date) as row_num,product_cat,date,sum(quantity) as quantity
FROM OCF.master_table 
where date > '2022-01-01'
group by product_cat,date
"""
```


```python
test_og = pd.read_sql(test_og, con = engine)
```


```python
test['date'] = pd.to_datetime(test['date']).dt.date
```


```python
parts = test["date"].astype(str).str.split("-", n = 3, expand = True)
test["year"]= parts[0].astype('int')
test["month"]= parts[1].astype('int')
test["day"]= parts[2].astype('int')
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_num</th>
      <th>product_cat</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>oyster shell</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>winter rye</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>greensand</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20-20-20</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>tube_tap</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = pd.get_dummies(test, columns = ['product_cat'])
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_num</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>product_cat_20-20-20</th>
      <th>product_cat_alfalfa</th>
      <th>product_cat_aluminium sulfate</th>
      <th>product_cat_bloodmeal</th>
      <th>product_cat_buckwheat</th>
      <th>...</th>
      <th>product_cat_damaged/return</th>
      <th>product_cat_flowers</th>
      <th>product_cat_garden staples</th>
      <th>product_cat_greensand</th>
      <th>product_cat_oyster shell</th>
      <th>product_cat_potatoes</th>
      <th>product_cat_rock phosphate</th>
      <th>product_cat_tube_tap</th>
      <th>product_cat_urea</th>
      <th>product_cat_winter rye</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
test['weekend'] = test.apply(lambda x:weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_num</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>product_cat_20-20-20</th>
      <th>product_cat_alfalfa</th>
      <th>product_cat_aluminium sulfate</th>
      <th>product_cat_bloodmeal</th>
      <th>product_cat_buckwheat</th>
      <th>...</th>
      <th>product_cat_flowers</th>
      <th>product_cat_garden staples</th>
      <th>product_cat_greensand</th>
      <th>product_cat_oyster shell</th>
      <th>product_cat_potatoes</th>
      <th>product_cat_rock phosphate</th>
      <th>product_cat_tube_tap</th>
      <th>product_cat_urea</th>
      <th>product_cat_winter rye</th>
      <th>weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
test['m1'] = np.sin(test['month'] * (2 * np.pi / 12))
test['m2'] = np.cos(test['month'] * (2 * np.pi / 12))
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_num</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>product_cat_20-20-20</th>
      <th>product_cat_alfalfa</th>
      <th>product_cat_aluminium sulfate</th>
      <th>product_cat_bloodmeal</th>
      <th>product_cat_buckwheat</th>
      <th>...</th>
      <th>product_cat_greensand</th>
      <th>product_cat_oyster shell</th>
      <th>product_cat_potatoes</th>
      <th>product_cat_rock phosphate</th>
      <th>product_cat_tube_tap</th>
      <th>product_cat_urea</th>
      <th>product_cat_winter rye</th>
      <th>weekend</th>
      <th>m1</th>
      <th>m2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2022-01-03</td>
      <td>2022</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.866025</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
test['weekday'] = test.apply(lambda x: which_day(x['year'],
                                                      x['month'],
                                                      x['day']),
                                   axis=1)

```


```python
test = test[[ 'row_num','month', 'day', 'weekend', 'm1', 'm2',
       'weekday', 'product_cat_20-20-20', 'product_cat_alfalfa',
       'product_cat_aluminium sulfate', 'product_cat_buckwheat',
       'product_cat_corn gluten', 'product_cat_flowers',
       'product_cat_garden staples', 'product_cat_greensand',
       'product_cat_oyster shell', 'product_cat_tube_tap',
       'product_cat_urea', 'product_cat_winter rye']]
```


```python
test.set_index("row_num", inplace = True)

```


```python
test = scaler.fit_transform(test)

Pred_Y = model.predict(test)
predict_time = time()-start    
print(model)
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

print(Pred_Y)
print(Pred_Y.size)
```

    Ridge()
    	Training time: 0.000s
    	Prediction time: 28.867s
    [ 5.70915341  8.67797242  8.07812575 ... -3.49274469 -1.81970771
     -4.40201979]
    1326



```python
test_og['predicted_quant'] = pd.DataFrame(Pred_Y)
```


```python
predicted = test_og.set_index(['date'])
predicted = predicted.groupby(['date'])[['quantity','predicted_quant']].sum()
```


```python
#Rolling average for predicted vs actual  
predicted.predicted_quant.rolling(10).mean().plot(figsize = (100,70),linewidth=10)
predicted.quantity.rolling(10).mean().plot(figsize = (100,70),linewidth=10)
plt.xlabel('Time', fontsize=90)
plt.ylabel('Predicted Inventory', fontsize=90)
plt.xticks(ha='right', rotation=55, fontsize=70, fontname='monospace')
plt.yticks(rotation=55, fontsize=70, fontname='monospace')
plt.title('2022 Predicted Output', fontsize=100)
plt.legend(loc=2,prop={'size': 75})
```




    <matplotlib.legend.Legend at 0x7fe6145e96d0>




    
![png](output_50_1.png)
    



```python

```
