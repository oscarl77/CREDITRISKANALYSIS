from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
import os.path

class CreditRiskAnalyser():
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()

    def analyse_credit_risk(self, new_file_path):
        raw_data = self.data_loader._load_data(self.file_path)
        if not os.path.isfile(new_file_path):
            preprocessed_file = self.data_preprocessor.preprocess(raw_data)
        preprocessed_file = new_file_path
        preprocessed_data = self.data_loader._load_data(preprocessed_file)