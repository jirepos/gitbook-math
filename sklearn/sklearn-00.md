# 사이킷런 데이터셋 사용

붓꽃 데이터 세트는 분류(Classification)를 하기 위한 대표적 데이터셋으로, 꽃잎의 길이, 너비를 기반으로 꽃의 품종을 예측할 수 있다.

![](../.gitbook/assets/sklearn/sklearn01.png)



> petal : 꽃잎  ,   sepal : 꽃받침    


## 데이터 가져오기

```python 
# iris 붓꽃 데이터 로드
from sklearn.datasets import load_iris
```
```python 
iris = load_iris()
iris
```
```
{'DESCR': '.. _iris_dataset:\n\nIris plants dataset\n--------------------\n\n**
...중략...
 'data': array([[5.1, 3.5, 1.4, 0.2],
        [4.9, 3. , 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5. , 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5. , 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
...생략...
```        


```python 
print(dir(iris))
# dir() 함수로 해당 객체의 변수와 메소드 확인
```
```
['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
```

* data : 피처의 데이터 셋
* target : 분류시 레이블 값, 회귀시 숫자 결과값 데이터
* target_names : 개별 레이블 이름 
* feature_names : 피처들의 이름 
* DESCR : 데이터 셋에 대한 설명과 각 피처에 대한 설명 

```python
iris.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```
```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

## 데이터 형태

```python
iris_data = iris.data
print(iris_data.shape) 
# shape써서 data의 행렬을 보여줌
# 150개의 데이터가 4개의 정보를 갖고 있음
# (150, 4)
```
```
(150, 4)
```


## 데이터의 인덱스 값 확인
```python
iris_data[0]
# data의 0번째 인덱스 값
# array([5.1, 3.5, 1.4, 0.2])
```
```
array([5.1, 3.5, 1.4, 0.2])
```

## 데이터 라벨링
```python 
iris_label = iris.target
iris_label
# 호출할 때 target으로 호출함
```
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

## 데이터 라벨링 이름

```python 
iris.target_names 
# target_names를 통해 데이터 라벨링 이름을 알 수 있다
```
```
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

* target : 정답(label)
* target_names : 정답 이름( setosa, versicolor, virginica)


> 레이블(=클래스, =타겟(값), =결정(값))지도학습에서 데이터의 학습을 위해 사용되는 정답 데이터


## 데이터셋 설명

```python
print(iris.DESCR)
# DESCR 함수를 통해 데이터셋의 설명을 볼 수 있다
```
```
**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    ...생략...
```




## 데이터 feature 설명
```python
iris.feature_names
```
```
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
```
이것은 데이터셋이 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃입 너비 등 4개의 컬럼으로 구성되어 있다는 것을 의미한다.


