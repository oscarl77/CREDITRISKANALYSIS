import pandas as pd
import os.path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class DataPreprocessor():

    def preprocess(self, raw_data):
        """Cleans and encodes data to be trained by Random Forest ML model.

        Args:
            raw_data (file): .csv file containing the raw unprocessed data.
        """
        cleaned_dataset = self._clean_dataset(raw_data)

        encoded_dataset = self._encode(cleaned_dataset)

        new_file_path = 'cleaned_credit_risk.csv'
        if os.path.isfile(new_file_path):
            return
        encoded_dataset.to_csv(new_file_path, index=False)


    def _clean_dataset(self, raw_data):
        cleaned_dataset = self._remove_incomplete_rows(raw_data)
        cleaned_dataset = self._remove_id_row(cleaned_dataset)
        return cleaned_dataset

    def _remove_incomplete_rows(self, raw_data):
        """Returns a the raw dataset without any rows containing missing
        values.

        Args:
            raw_data (file): .csv file containing the raw credit risk dataset.

        Returns:
            file: .csv file without the rows containing missing values.
        """
        for column in raw_data.columns:
            if raw_data[column].isnull().any():
                raw_data = raw_data[raw_data[column].notna()]
        return raw_data
    
    def _remove_id_row(self, dataset):
        dataset = dataset.drop(columns=["Id"])
        return dataset
    

    def _encode(self, dataset):
        """Returns dataset with columns containing categorical values encoded.

        Args:
            dataset (file): .csv file containing dataset.

        Returns:
            file: .csv file containing the encoded dataset.
        """
        home_ownership_encoded = self._one_hot_encode(dataset, 'Home')
        intent_encoded = self._one_hot_encode(dataset, 'Intent')

        self._label_encode(dataset, 'Default')
        
        dataset = pd.concat([dataset, home_ownership_encoded, intent_encoded], axis=1).drop(columns=['Home', 'Intent'])
        return dataset
    
    def _one_hot_encode(self, dataset, column_heading):
        """Encodes given column in dataset using one-hot-encoding.

        Args:
            dataset (file): .csv file containing dataset
            column_heading (str): heading name of the column to be encoded.

        Returns:
            pandas.DataFrame: dataframe containing the newly encoded columns.
        """
        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
        oh_encoder_transform = oh_encoder.fit_transform(dataset[[column_heading]])
        return oh_encoder_transform
    
    def _label_encode(self, dataset, column_heading):
        """Encodes given column in dataset using label encoding.

        Args:
            dataset (file): .csv file containing dataset.
            column_heading (str): heading name of the column to be encoded.
        """
        lb_encoder = LabelEncoder()
        dataset[column_heading] = lb_encoder.fit_transform(dataset[[column_heading]])