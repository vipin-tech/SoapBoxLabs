# Filter Errorenous datapoints in the dataset

This application helps to filter out the errorenous data points in the dataset using two different algorithms.

## Prerequisites

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


Both these algorithms return the anomaly score for each sample which helps in differentiating between errorneous and non-errorneous datapoints.


## Analysis

### From the below plots we notice some errorenous data points (spikes) in the dataset.

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image01.png" alt="" width="800px" height="450px">

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image02.png" alt="" width="800px" height="450px">


### Detection of errorenous points using Isolation Forest algorithm.

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image03.png" alt="" width="800px" height="450px">

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image04.png" alt="" width="800px" height="450px">


### Detection of errorenous points using OneClassSVM algorithm.

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image10.png" alt="" width="800px" height="450px">

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image11.png" alt="" width="800px" height="450px">


## Application Output

### Non Errorenous data points found using Isolation Forest and OneClassSVM algorithms

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image05.png" alt="">

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image06.png" alt="">

### Errorenous data points found using IsolationForest and OneClassSVM algorithms

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image07.png" alt="">

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image08.png" alt="">

### Unit Tests

<img src="https://github.com/vipin-tech/SoapBoxLabs/blob/master/screenshots/Image09.png" alt="">


### Conclusion

_From the above plots and output, it is evident that Isolation Forest agorithm performs better as compared to the OneClassSVM algorithm in detecting errorenous points for this dataset._
