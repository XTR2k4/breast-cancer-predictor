import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def get_clean_data():
    data = pd.read_csv('D:\CodeSpace\Breast Cancer Diagnosis\data\data.csv')
    data = data.drop(['Unnamed: 32','id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0}) 
    return data

def create_model(data):
    X = data.drop(columns='diagnosis',axis=1)
    y = data['diagnosis']
    
    #scale the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    #training the model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    #test the model
    y_pred = model.predict(X_test)
    print(f'Accuracy of the model: {accuracy_score(y_true=y_test,y_pred=y_pred)}')
    print(f'Classification report: \n {classification_report(y_true=y_test,y_pred=y_pred)}')
    
    return model, sc

def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    with open('model/model.pkl','wb')as file:
        pickle.dump(model,file)
    with open('model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler,file)

if __name__== '__main__':
    main()