# Import the Modules.
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict


class OutlierDetector:
    """
    OutlierDetector class helps to load the dataset and get the summary
    statistics of the dataset.
    """

    def __init__(self, **kwargs: Dict[str, str]):
        self.path = kwargs.get('path')
        self.file = kwargs.get('file')
        self.df = None
        self.loadData()

    def loadData(self) -> None:
        """
        Load the dataset.
        """
        try:
            if self.path and self.file:
                file = self.path + '/' + self.file
                self.df = pd.read_csv(file, names=['Latitude', 'Longitude',
                                                   'Timestamp'])
            else:
                print('Dataset path not specified')

        except FileNotFoundError as ex:
            raise FileNotFoundError('File Not found in the specified path.\
                                    Error: {}'.format(str(ex)))
        except Exception as ex:
            raise Exception('Error while loading data points. {}'
                            .format(str(ex)))

    def getSampleData(self) -> None:
        """
        Get the first 5 sample records from the dataset.
        """

        if self.df is None:
            self.loadData()
        else:
            return self.df.head()

    def getMetadata(self) -> pd.DataFrame:
        """
        Get the description of the attributes in the dataset.
        """

        if self.df is None:
            self.loadData()
        else:
            return self.df.info()

    def describeDataAttribute(self, attr_name: str = None) -> pd.Series:
        """
        Description in detail based on the attribute name.
        :attr_name: Attribute name of the attribute
        """

        try:
            if attr_name:
                return self.df[attr_name].describe()
        except (KeyError, Exception) as ex:
            raise Exception('Exception in describeDataAttribute \
                            {}'.format(str(ex)))

    def scale(self) -> None:
        raise NotImplementedError

    def fit(self) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def compute(self) -> None:
        """
        Method to fit the model and predict the erroneous data points
        """
        try:
            self.scale()
            self.fit()
            self.predict()

        except KeyError as ex:
            raise KeyError('Trying to access invalid key. Error: {}'.
                           format(str(ex)))

        except NotImplementedError:
            raise NotImplementedError('Not Implemented scale/fit/predict methods.')

        except Exception as ex:
            raise Exception('Error while building IsolationForestModel. Error: {}'.format(str(ex)))


class IsolationForestModel(OutlierDetector):
    """
    IsolationForestModel class helps to identify the erroneous points in the
    dataset based on the anomaly score computed using IsolationForest Algorithm
    """

    def __init__(self, outlier_fraction=0.03, **kwargs):
        OutlierDetector.__init__(self, **kwargs)
        self.outlier_fraction = outlier_fraction
        self.scale_data = None
        self.err_points = None
        self.model = None
        self.compute()

    def scale(self) -> None:
        """
        Method to scale the attributes.
        """

        data = self.df[['Latitude', 'Longitude']]
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data)
        self.scale_data = pd.DataFrame(np_scaled)

    def fit(self) -> None:
        """
        Method to fit the model on the scaled data.
        """

        self.model = IsolationForest(contamination=self.outlier_fraction)
        self.model.fit(self.scale_data)

    def predict(self) -> None:
        """
        Method to predict the erroneous data points on the fitted model.
        """

        self.df['ifm_anomaly'] = pd.Series(self.model.predict(self.scale_data))

    def getErrorneousPoints(self) -> pd.DataFrame:
        """
        Method to filter the erroneous points based on the anomaly score.
        """
        return self.df.loc[self.df['ifm_anomaly'] == -1, ['Timestamp',
                                                          'Latitude',
                                                          'Longitude']]

    def getDataPoints(self) -> pd.DataFrame:
        """
        This method returns non-erroneous data points .
        """
        try:
            return self.df.loc[self.df['ifm_anomaly'] == 1, ['Timestamp',
                                                             'Latitude',
                                                             'Longitude']]
        except KeyError as ex:
            raise KeyError('Trying to access invalid key. Error: {}'.
                           format(ex))
        except Exception as ex:
            raise Exception('Error while fetching Non-Errorneous data points. Error: {}'.format(str(ex)))


class OneClassSVMModel(OutlierDetector):
    """
    OneClassSVMModel class helps to identify the erroneous points in the
    dataset based on the anomaly score computed using OneClassSVM Algorithm
    """

    def __init__(self, outlier_fraction=0.03, kernel='rbf', gamma=0.01,
                 **kwargs):
        OutlierDetector.__init__(self, **kwargs)
        self.outlier_fraction = outlier_fraction
        self.scale_data = None
        self.err_points = None
        self.model = None
        self.kernel = kernel
        self.gamma = gamma
        self.compute()

    def scale(self) -> None:
        """
        Method to scale the attributes.
        """

        data = self.df[['Latitude', 'Longitude']]
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data)
        self.scale_data = pd.DataFrame(np_scaled)

    def fit(self) -> None:
        """
        Method to fit the model on the scaled data.
        """

        self.model = OneClassSVM(nu=self.outlier_fraction, kernel=self.kernel,
                                 gamma=self.gamma)
        self.model.fit(self.scale_data)

    def predict(self) -> None:
        """
        Method to predict the errorneous points on the fitted model.
        """

        self.df['ocs_anomaly'] = pd.Series(self.model.predict(self.scale_data))

    def getErrorneousPoints(self) -> pd.DataFrame:
        """
        Method to filter the errorneous points based on the anomaly score.
        """
        try:
            return self.df.loc[self.df['ocs_anomaly'] == -1, ['Timestamp',
                                                              'Latitude',
                                                              'Longitude']]
        except KeyError as ex:
            raise KeyError('Trying to access invalid key. Error: {}'.
                           format(ex))
        except Exception as ex:
            raise Exception('Error while fetching Errorneous data points. Error: {}'.format(str(ex)))

    def getDataPoints(self) -> pd.DataFrame:
        """
        This method returns non-erroneous data points .
        """
        return self.df.loc[self.df['ocs_anomaly'] == 1, ['Timestamp',
                                                         'Latitude',
                                                         'Longitude']]


class Client:
    """
    Method to in invoke the different algorithms to identify erroneous
    data points.
    """

    def __init__(self, **kwargs: Dict[str, str]):
        self.model = None
        self.model_name = None
        self.kwargs = kwargs

    def buildModel(self, model_name: str = None) -> None:
        """
        Method to build model based on model name.
        :model_name: Name of the Algorithm to be used to identify erroneous
        data points.
        """
        model_name = model_name.lower()
        if model_name not in ['oneclasssvm', 'isolationforest']:
            raise Exception('Invalid model name specified.')
            return

        self.model_name = model_name
        if model_name == 'oneclasssvm':
            self.model = OneClassSVMModel(**self.kwargs)

        elif model_name == 'isolationforest':
            self.model = IsolationForestModel(**self.kwargs)

    def getErrorneousDataPoints(self) -> pd.DataFrame:
        """
        Method to get the errorneous data points.
        """
        if self.model:
            print('Erroneous points in the dataset found using model {}:'
                  .format(self.model_name))
            return self.model.getErrorneousPoints()
        else:
            print('Model Not Initialised')

    def getDataPoints(self) -> pd.DataFrame:
        """
        Method to get non-errorneous data points.
        """
        if self.model:
            print('Non-Errorneous points in the dataset found using model {}'
                  .format(self.model_name))
            return self.model.getDataPoints()
        else:
            print('Model not Initialised')


# Main Program
if __name__ == '__main__':

    path = input('Please enter the path where data points are stored: ').strip()
    file = input('Please enter the file name: ').strip()
    try:
        client = Client(path=path, file=file)
        model_name = input('Please enter the option: 1. Isolation Forest 2. OneClassSVM \n').strip()

        algorithms = {'1': 'isolationforest', '2': 'oneclasssvm'}
        algorithm = algorithms.get(model_name)
        if algorithm:
            client.buildModel(algorithm)
        else:
            print('Invalid Option specified.')

        # Need to return non-errorneous data points
        # errorneous_points = client.getErrorneousDataPoints()
        data_points = client.getDataPoints()

        # print(errorneous_points)
        print(data_points)

    # @vipin (enchancement): Log the Exceptions.
    except FileNotFoundError as ex:
        print(str(ex))

    except KeyError as ex:
        print(str(ex))

    except NotImplementedError as ex:
        print(str(ex))

    except Exception as ex:
        print(str(ex))
