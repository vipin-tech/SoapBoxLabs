# Filter Errorenous datapoints in the dataset

This application helps to filter out the errorenous data points in the dataset using two different algorithms.

## Prerequites

* Python 3.x
* numpy
* pandas
* scikit-learn

> **How to run the application?**
>> python application.py

> **How to run the test cases?**
>> python test_application.py

## Algorithms

The two algorithms used are:

1. Isolation Forest
2. OneClassSVM


### Isolation Forest

The IsolationForest Algorithm ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

### OneClassSVM

The OneClassSVM Algorithm helps in the detection of the outliers in the dataset. The implementation of this algorithm is based on libsvm.


Both these algorithms return the anomaly score of each sample using the IsolationForest algorithm.


## Analysis

### From the below plots we notice some errorenous data points (spikes) in the dataset.

![Timestamp vs Latitude](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image01.png)


![Timestamp vs Latitude](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image02.png)

### Detection of errorenous points using Isolation Forest algorithm.

![Timestamp vs Latitude](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image03.png)

![Timestamp vs Latitude](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image04.png)


## Application Output

### Non Errorenous data points found using Isolation Forest and OneClassSVM algorithms

![IsolationForest](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image05.png)

![OneClassSVM](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image06.png)

### Errorenous data points found using IsolationForest and OneClassSVM algorithms

![IsolationForest](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image07.png)

![OneClassSVM](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image08.png)

### Unit Tests

![Unit Test](https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image09.png)


### Summary

From the above plots and output, it is evident that Isolation Forests perform better as compared to the OneClassSVM algorithm in detecting errorenous points for  this dataset.
