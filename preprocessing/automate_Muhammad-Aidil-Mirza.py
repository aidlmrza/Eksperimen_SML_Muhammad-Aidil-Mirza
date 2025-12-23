import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
    return outliers

def preprocess_data(df):
    if df.isnull().sum().any():
        df.dropna(inplace=True)
        print("Missing values dihapus.")
    
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        print("Duplikasi dihapus.")
    
    outliers_Pregnancies = detect_outliers_iqr(df['Pregnancies'])
    outliers_Glucose = detect_outliers_iqr(df['Glucose'])
    outliers_BloodPressure = detect_outliers_iqr(df['BloodPressure'])
    outliers_SkinThickness = detect_outliers_iqr(df['SkinThickness'])
    outliers_Insulin = detect_outliers_iqr(df['Insulin'])
    outliers_BMI = detect_outliers_iqr(df['BMI'])
    outliers_DiabetesPedigreeFunction = detect_outliers_iqr(df['DiabetesPedigreeFunction'])
    outliers_Age = detect_outliers_iqr(df['Age'])

    all_outliers = list(set(outliers_Pregnancies + outliers_Glucose + outliers_BloodPressure + outliers_SkinThickness + outliers_Insulin + outliers_BMI + outliers_DiabetesPedigreeFunction + outliers_Age))

    print(f"Jumlah outlier yang terdeteksi: {len(all_outliers)}")

    df.drop(index=all_outliers, inplace=True)

    return df


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data yang sudah diproses disimpan di {output_path}")

if __name__ == "__main__":
    raw_data_path = './diabetes_raw.csv'
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    df_preprocessed = pd.DataFrame(X, columns=X.columns)
    df_preprocessed['Outcome'] = y.reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    processed_data_path = './preprocessing/diabetes_preprocessing.csv' 
    save_preprocessed_data(df_preprocessed, processed_data_path)
