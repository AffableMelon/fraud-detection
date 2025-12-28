import unittest
import os
import sys
import pandas as pd
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import FraudDataProcessor

class TestProjectStructure(unittest.TestCase):
    def test_directories_exist(self):
        """Test that key directories exist"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        required_dirs = ['data', 'notebooks', 'scripts', 'src', 'models', 'tests', 'reports']
        for d in required_dirs:
            self.assertTrue(os.path.exists(os.path.join(base_dir, d)), f"Directory {d} missing")

    def test_scripts_exist(self):
        """Test that key scripts exist"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        scripts = [
            'scripts/task2_model_training.py',
            'src/data_loader.py'
        ]
        for s in scripts:
            self.assertTrue(os.path.exists(os.path.join(base_dir, s)), f"Script {s} missing")

class TestFraudDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FraudDataProcessor(
            fraud_path="dummy_fraud.csv",
            ip_path="dummy_ip.csv",
            credit_path="dummy_credit.csv"
        )

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_data_success(self, mock_exists, mock_read_csv):
        """Test successful data loading"""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        self.processor.load_data()
        
        self.assertIsNotNone(self.processor.df_fraud)
        self.assertIsNotNone(self.processor.df_ip)
        self.assertIsNotNone(self.processor.df_credit)

    @patch('os.path.exists')
    def test_load_data_file_not_found(self, mock_exists):
        """Test error handling when file is missing"""
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.processor.load_data()

if __name__ == '__main__':
    unittest.main()
