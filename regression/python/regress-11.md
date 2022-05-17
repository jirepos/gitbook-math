# statsmodel 다중선형회귀분석



## 데이터 살펴보기
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
```


```python
df = pd.DataFrame({
'radio_ads': [3,4,9,4,5,5,2,6,5,3],
'tv_ads':    [1,3,4,1,4,1,4,2,4,2],
'retention': [5,1,6,2,8,3,4,9,7,4],
})

df
```
```
	radio_ads	tv_ads	retention
0	3	1	5
1	4	3	1
2	9	4	6
3	4	1	2
4	5	4	8
5	5	1	3
6	2	4	4
7	6	2	9
8	5	4	7
9	3	2	4
```
```python
X = df[['radio_ads','tv_ads']]   # 독립변수 분리
y = df['retention']   # 종속변수 분리
X = sm.add_constant(X)  # 상수항 결합
X
```
```
const	radio_ads	tv_ads
0	1.0	3	1
1	1.0	4	3
2	1.0	9	4
3	1.0	4	1
4	1.0	5	4
5	1.0	5	1
6	1.0	2	4
7	1.0	6	2
8	1.0	5	4
9	1.0	3	2
```
