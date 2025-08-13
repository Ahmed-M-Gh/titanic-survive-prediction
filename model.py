#Importing Modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib as jb

#reading file
df = pd.read_csv(r'C:\I will prepare my self to destroy the world\Programming\python program\MY Projects\Classification Projects\1 - Titanik Survive Prediction app\titanic-train.csv')

#Deleting unimportant data
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Handling null
df = df[df['Embarked'].notnull()]
median_Age = df['Age'].median()
df['Age'].fillna(median_Age, inplace=True)

#Handling Outliers
Outlier_list = ['Age', 'SibSp', 'Parch', 'Fare']
def remove_outliers_iqr(data, columns):
    cleaned_df = data.copy()
    for column in columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1
        Lower = Q1 - IQR * 1.5
        Upper = Q3 + IQR * 1.5
        cleaned_df = cleaned_df[(cleaned_df[column] >= Lower) & (cleaned_df[column] <= Upper)]
    return cleaned_df
clean_df = remove_outliers_iqr(df, Outlier_list)


#Encoding
LE = LabelEncoder()
clean_df['Sex'] = LE.fit_transform(clean_df['Sex'])
OE = OneHotEncoder(sparse_output=False)
embarked_encoded = pd.DataFrame(OE.fit_transform(clean_df[['Embarked']]), columns=OE.get_feature_names_out(['Embarked']))
clean_df.reset_index(drop=True, inplace=True)
embarked_encoded.reset_index(drop=True, inplace=True)
df = pd.concat([clean_df.drop(['Embarked'], axis=1), embarked_encoded], axis=1)

#Split DF for X and Y
X = df.drop('Survived', axis=1)
Y = df['Survived']

#Split Data to Train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

#Scaling X values to make it between 1, 0
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)

#----------------------------------------------------------------------

#train the model by the 
model = KNeighborsClassifier()
model.fit(x_train_scaled, y_train)

#Saving resluts
saving_path = r'C:\I will prepare my self to destroy the world\Programming\python program\MY Projects\Classification Projects\1 - Titanik Survive Prediction\\'
jb.dump(model , saving_path + 'model_train_result.sav')
jb.dump(scale , saving_path + 'Standared_Scalar.sav')
jb.dump(LE , saving_path + 'Label_encoding_sex.sav')
jb.dump(OE , saving_path + 'One_Hot_encoding_embarked.sav')
jb.dump(list(X.columns) , saving_path + 'column_encoded_result.sav')