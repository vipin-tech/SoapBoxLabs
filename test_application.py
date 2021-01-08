import unittest
import pandas as pd
from application import Client, OneClassSVMModel, IsolationForestModel
import os


class TestClient(unittest.TestCase):

    def test_client_instance(self):
        path, file = 'fake_path', 'fake_file'
        client = Client(path=path, file=file)
        self.assertIsInstance(client, Client)

    def test_invalid_path(self):
        path, file = 'fake_path', 'test_data_points.csv'
        client = Client(path=path, file=file)
        with self.assertRaises(FileNotFoundError):
            client.buildModel('OneClassSVM')

    def test_invalid_file(self):
        path, file = os.getcwd(), 'fake_file'
        client = Client(path=path, file=file)
        with self.assertRaises(FileNotFoundError):
            client.buildModel('OneClassSVM')

    def test_build_model_instance(self):
        path, file = os.getcwd(), 'test_data_points.csv'
        client = Client(path=path, file=file)
        client.buildModel('OneClassSVM')
        self.assertIsInstance(client.model, OneClassSVMModel)

        client.buildModel('IsolatedForest')
        self.assertIsInstance(client.model, IsolationForestModel)

    def test_invalid_model_name(self):
        path, file = os.getcwd(), 'test_data_points.csv'
        client = Client(path=path, file=file)
        with self.assertRaises(Exception):
            client.buildModel('fake_model')

    def test_get_data_points(self):
        path, file = os.getcwd(), 'test_data_points.csv'
        client = Client(path=path, file=file)
        client.buildModel('OneClassSVM')

        data_points = client.getDataPoints()
        self.assertIsInstance(data_points, pd.DataFrame)

    def test_get_errorenous_data_points(self):
        path, file = os.getcwd(), 'test_data_points.csv'
        client = Client(path=path, file=file)
        client.buildModel('OneClassSVM')

        err_data_points = client.getErrorneousDataPoints()
        self.assertIsInstance(err_data_points, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
