```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
% matplotlib inline
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_24140/2549244765.py in <module>
    ----> 1 import pandas as pd
          2 import numpy as np
          3 import matplotlib.pyplot as pit
          4 get_ipython().run_line_magic('', 'matplotlib inline')
    

    ModuleNotFoundError: No module named 'pandas'



```python
import pandas
import numpy as np
import matplotlib.pyplot as pit
% matplotlib inline
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_24140/4268495776.py in <module>
    ----> 1 import pandas
          2 import numpy as np
          3 import matplotlib.pyplot as pit
          4 get_ipython().run_line_magic('', 'matplotlib inline')
    

    ModuleNotFoundError: No module named 'pandas'



```python
pip install pandas
```

    Collecting pandas
      Downloading pandas-1.3.3-cp39-cp39-win_amd64.whl (10.2 MB)
    Collecting numpy>=1.17.3
      Downloading numpy-1.21.2-cp39-cp39-win_amd64.whl (14.0 MB)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from pandas) (2.8.2)
    Collecting pytz>=2017.3
      Downloading pytz-2021.1-py2.py3-none-any.whl (510 kB)
    Requirement already satisfied: six>=1.5 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)
    Installing collected packages: pytz, numpy, pandas
    Successfully installed numpy-1.21.2 pandas-1.3.3 pytz-2021.1
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: You are using pip version 21.2.3; however, version 21.2.4 is available.
    You should consider upgrading via the 'C:\Users\shrik\AppData\Local\Programs\Python\Python39\python.exe -m pip install --upgrade pip' command.
    


```python
!pip install pandas
```

    Requirement already satisfied: pandas in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (1.3.3)
    Requirement already satisfied: pytz>=2017.3 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from pandas) (2021.1)
    Requirement already satisfied: numpy>=1.17.3 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from pandas) (1.21.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)
    

    WARNING: You are using pip version 21.2.3; however, version 21.2.4 is available.
    You should consider upgrading via the 'C:\Users\shrik\AppData\Local\Programs\Python\Python39\python.exe -m pip install --upgrade pip' command.
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
% matplotlib inline
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_24140/2549244765.py in <module>
          1 import pandas as pd
          2 import numpy as np
    ----> 3 import matplotlib.pyplot as pit
          4 get_ipython().run_line_magic('', 'matplotlib inline')
    

    ModuleNotFoundError: No module named 'matplotlib'



```python
pip install matplotlib
```

    Collecting matplotlib
      Downloading matplotlib-3.4.3-cp39-cp39-win_amd64.whl (7.1 MB)
    Note: you may need to restart the kernel to use updated packages.
    

    ERROR: Exception:
    Traceback (most recent call last):
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\urllib3\response.py", line 438, in _error_catcher
        yield
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\urllib3\response.py", line 519, in read
        data = self._fp.read(amt) if not fp_closed else b""
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\cachecontrol\filewrapper.py", line 62, in read
        data = self.__fp.read(amt)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\http\client.py", line 462, in read
        n = self.readinto(b)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\http\client.py", line 506, in readinto
        n = self.fp.readinto(b)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\socket.py", line 704, in readinto
        return self._sock.recv_into(b)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\ssl.py", line 1241, in recv_into
        return self.read(nbytes, buffer)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\ssl.py", line 1099, in read
        return self._sslobj.read(len, buffer)
    ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\cli\base_command.py", line 173, in _main
        status = self.run(options, args)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\cli\req_command.py", line 203, in wrapper
        return func(self, options, args)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\commands\install.py", line 315, in run
        requirement_set = resolver.resolve(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\resolver.py", line 94, in resolve
        result = self._result = resolver.resolve(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 472, in resolve
        state = resolution.resolve(requirements, max_rounds=max_rounds)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 341, in resolve
        self._add_to_criteria(self.state.criteria, r, parent=None)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\resolvelib\resolvers.py", line 172, in _add_to_criteria
        if not criterion.candidates:
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\resolvelib\structs.py", line 151, in __bool__
        return bool(self._sequence)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 140, in __bool__
        return any(self)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 128, in <genexpr>
        return (c for c in iterator if id(c) not in self._incompatible_ids)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\found_candidates.py", line 32, in _iter_built
        candidate = func()
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\factory.py", line 204, in _make_candidate_from_link
        self._link_candidate_cache[link] = LinkCandidate(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 295, in __init__
        super().__init__(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 156, in __init__
        self.dist = self._prepare()
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 227, in _prepare
        dist = self._prepare_distribution()
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\resolution\resolvelib\candidates.py", line 305, in _prepare_distribution
        return self._factory.preparer.prepare_linked_requirement(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\operations\prepare.py", line 508, in prepare_linked_requirement
        return self._prepare_linked_requirement(req, parallel_builds)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\operations\prepare.py", line 550, in _prepare_linked_requirement
        local_file = unpack_url(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\operations\prepare.py", line 239, in unpack_url
        file = get_http_url(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\operations\prepare.py", line 102, in get_http_url
        from_path, content_type = download(link, temp_dir.path)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\network\download.py", line 145, in __call__
        for chunk in chunks:
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\cli\progress_bars.py", line 144, in iter
        for x in it:
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_internal\network\utils.py", line 63, in response_chunks
        for chunk in response.raw.stream(
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\urllib3\response.py", line 576, in stream
        data = self.read(amt=amt, decode_content=decode_content)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\urllib3\response.py", line 541, in read
        raise IncompleteRead(self._fp_bytes_read, self.length_remaining)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\contextlib.py", line 137, in __exit__
        self.gen.throw(typ, value, traceback)
      File "C:\Users\shrik\AppData\Local\Programs\Python\Python39\lib\site-packages\pip\_vendor\urllib3\response.py", line 455, in _error_catcher
        raise ProtocolError("Connection broken: %r" % e, e)
    pip._vendor.urllib3.exceptions.ProtocolError: ("Connection broken: ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None)", ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    WARNING: You are using pip version 21.2.3; however, version 21.2.4 is available.
    You should consider upgrading via the 'C:\Users\shrik\AppData\Local\Programs\Python\Python39\python.exe -m pip install --upgrade pip' command.
    


```python
pip install --upgrade pip
```

    Requirement already satisfied: pip in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (21.2.3)
    Collecting pip
      Downloading pip-21.2.4-py3-none-any.whl (1.6 MB)
    Installing collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 21.2.3
        Uninstalling pip-21.2.3:
          Successfully uninstalled pip-21.2.3
    Successfully installed pip-21.2.4
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
% matplotlib inline
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_24140/2549244765.py in <module>
          1 import pandas as pd
          2 import numpy as np
    ----> 3 import matplotlib.pyplot as pit
          4 get_ipython().run_line_magic('', 'matplotlib inline')
    

    ModuleNotFoundError: No module named 'matplotlib'



```python
pip install matplotlib
```

    Collecting matplotlib
      Downloading matplotlib-3.4.3-cp39-cp39-win_amd64.whl (7.1 MB)
    Collecting kiwisolver>=1.0.1
      Downloading kiwisolver-1.3.2-cp39-cp39-win_amd64.whl (52 kB)
    Collecting cycler>=0.10
      Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from matplotlib) (2.4.7)
    Collecting pillow>=6.2.0
      Downloading Pillow-8.3.2-cp39-cp39-win_amd64.whl (3.2 MB)
    Requirement already satisfied: numpy>=1.16 in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from matplotlib) (1.21.2)
    Requirement already satisfied: six in c:\users\shrik\appdata\local\programs\python\python39\lib\site-packages (from cycler>=0.10->matplotlib) (1.16.0)
    Installing collected packages: pillow, kiwisolver, cycler, matplotlib
    Successfully installed cycler-0.10.0 kiwisolver-1.3.2 matplotlib-3.4.3 pillow-8.3.2
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
% matplotlib inline
```

    UsageError: Line magic function `%` not found.
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
%matplotlib inline
```


```python
df = pd.read_csv("creditcard.csv")
```


```python
df.head(10)
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
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
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
## Dataset 4: Rotten Tomatoes Reviews
Link: [Google Dataset] (https://drive.google.com/file/d/1w1TsJB-gmIkZ28d1j7sf1sqcPmHXw352/view)
Purpose: I want to segregate the fresh and rotten reviews
## Dataset 5: Wine Quality Dataset
Link: [U.C.I.] (https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
Purpose: I want to look into different chemical information about the wine
'''
```




    '\n## Dataset 1: Credit Card Fraud Detection Dataset\nLink: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)\nPurpose: I want to use this dataset to detect fraudulent activities.\n## Dataset 2: IMDB Dataset\nLink: [Kaggle Dataset] (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)\nPurpose: I want to predict the number of positive and negative reviews from this dataset.\n## Dataset 3: Mall Customers\nLink: [Kaggle Dataset] (https://www.kaggle.com/shwetabh123/mall-customers)\nPurpose: I want to use this database to segment customers based on their age, income, and interest.\n## Dataset 4: Rotten Tomatoes Reviews\nLink: [Google Dataset] (https://drive.google.com/file/d/1w1TsJB-gmIkZ28d1j7sf1sqcPmHXw352/view)\nPurpose: I want to segregate the fresh and rotten reviews\n## Dataset 5: Wine Quality Dataset\nLink: [U.C.I.] (https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)\nPurpose: I want to look into different chemical information about the wine\n'




```python

```
