```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set()

df = pd.read_csv("creditcard.csv")
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
'''
## Dataset 1: Credit Card Fraud Detection Dataset
Link: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
Purpose: I want to use this dataset to detect fraudulent activities.
## Dataset 2: IMDB Dataset
Link: [Kaggle Dataset] (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
Purpose: I want to predict the number of positive and negative reviews from this dataset.
## Dataset 3: Mall Customers
Link: [Kaggle Dataset] (https://www.kaggle.com/shwetabh123/mall-customers)
Purpose: I want to use this database to segment customers based on their age, income, and interest.
## Dataset 4: Dataset Surgical Binary Classification
Link: [Kaggle Dataset] (https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification)
Purpose: I want to use the digital data that is in binary dataset
## Dataset 5: Wine Quality Dataset
Link: [U.C.I.] (https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
Purpose: I want to look into different chemical information about the wine
'''
```




    '\n## Dataset 1: Credit Card Fraud Detection Dataset\nLink: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)\nPurpose: I want to use this dataset to detect fraudulent activities.\n## Dataset 2: IMDB Dataset\nLink: [Kaggle Dataset] (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)\nPurpose: I want to predict the number of positive and negative reviews from this dataset.\n## Dataset 3: Mall Customers\nLink: [Kaggle Dataset] (https://www.kaggle.com/shwetabh123/mall-customers)\nPurpose: I want to use this database to segment customers based on their age, income, and interest.\n## Dataset 4: Dataset Surgical Binary Classification\nLink: [Kaggle Dataset] (https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification)\nPurpose: I want to use the digital data that is in binary dataset\n## Dataset 5: Wine Quality Dataset\nLink: [U.C.I.] (https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)\nPurpose: I want to look into different chemical information about the wine\n'




```python
df = pd.read_csv("IMDB Dataset.csv")
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
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv("Mall_Customers.csv")
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
      <th>CustomerID</th>
      <th>Genre</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv("Surgical-deepnet.csv")
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
      <th>bmi</th>
      <th>Age</th>
      <th>asa_status</th>
      <th>baseline_cancer</th>
      <th>baseline_charlson</th>
      <th>baseline_cvd</th>
      <th>baseline_dementia</th>
      <th>baseline_diabetes</th>
      <th>baseline_digestive</th>
      <th>baseline_osteoart</th>
      <th>...</th>
      <th>complication_rsi</th>
      <th>dow</th>
      <th>gender</th>
      <th>hour</th>
      <th>month</th>
      <th>moonphase</th>
      <th>mort30</th>
      <th>mortality_rsi</th>
      <th>race</th>
      <th>complication</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.31</td>
      <td>59.2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-0.57</td>
      <td>3</td>
      <td>0</td>
      <td>7.63</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>-0.43</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.73</td>
      <td>59.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.21</td>
      <td>0</td>
      <td>0</td>
      <td>12.93</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.41</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.85</td>
      <td>59.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>7.68</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0.08</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.49</td>
      <td>59.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>-0.65</td>
      <td>2</td>
      <td>1</td>
      <td>7.58</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>-0.32</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.70</td>
      <td>59.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>7.88</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df = pd.read_csv("winequality-red.csv")
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
      <th>fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv("Mall_Customers.csv")
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
      <th>CustomerID</th>
      <th>Genre</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# bar drawing function
def bar_chart(feature):
    Male = df[df['Genre']==1][feature].value_counts()
    Female = df[df['Genre']==0][feature].value_counts()
    df_bar = pd.DataFrame([Male,Female])
    df_bar.index = ['Male','Female']
    df_bar.plot(kind='bar',stacked=True, figsize=(7,5))
```


```python
# Spending Score  and  Genre ( Femal=0, Male=1)
bar_chart('Spending Score (1-100)')
plt.xlabel('Genre')
plt.ylabel('Spending Score')
plt.legend()
plt.title('Genre and Spending Score')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_10984/2159267155.py in <module>
          1 # Spending Score  and  Genre ( Femal=0, Male=1)
    ----> 2 bar_chart('Spending Score (1-100)')
          3 plt.xlabel('Genre')
          4 plt.ylabel('Spending Score')
          5 plt.legend()
    

    ~\AppData\Local\Temp/ipykernel_10984/738110016.py in bar_chart(feature)
          5     df_bar = pd.DataFrame([Male,Female])
          6     df_bar.index = ['Male','Female']
    ----> 7     df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
    

    ~\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\plotting\_core.py in __call__(self, *args, **kwargs)
        970                     data.columns = label_name
        971 
    --> 972         return plot_backend.plot(data, kind=kind, **kwargs)
        973 
        974     __call__.__doc__ = __doc__
    

    ~\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\plotting\_matplotlib\__init__.py in plot(data, kind, **kwargs)
         69             kwargs["ax"] = getattr(ax, "left_ax", ax)
         70     plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    ---> 71     plot_obj.generate()
         72     plot_obj.draw()
         73     return plot_obj.result
    

    ~\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\plotting\_matplotlib\core.py in generate(self)
        284     def generate(self):
        285         self._args_adjust()
    --> 286         self._compute_plot_data()
        287         self._setup_subplots()
        288         self._make_plot()
    

    ~\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\plotting\_matplotlib\core.py in _compute_plot_data(self)
        451         # no non-numeric frames or series allowed
        452         if is_empty:
    --> 453             raise TypeError("no numeric data to plot")
        454 
        455         self.data = numeric_data.apply(self._convert_to_ndarray)
    

    TypeError: no numeric data to plot



```python
plt.figure(figsize=(12,6))
sns.barplot(x='Genre',y='Spending Score (1-100)',data=df)
plt.title('')
plt.show()
```


    
![png](output_9_0.png)
    



```python
plt.figure(figsize=(12,6))
sns.countplot(y='Genre', data=df)
plt.title('Gender representation')
plt.show()
```


    
![png](output_10_0.png)
    



```python
for col in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    plt.figure(figsize=(12,6))
    sns.histplot(data=df,x=col, kde=True)
    plt.show()
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



```python
sns.set_theme(style="ticks")
sns.pairplot(df,hue='Genre',height=5)
plt.show()
```


    
![png](output_12_0.png)
    



```python

```
