import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle5 as pickle

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    # print(data.head())
    data = data.drop(['Unnamed: 32', 'id'],axis=1)
    data ['diagnosis'] = data['diagnosis'].map({'M': 1,'B':0})
    return data

def create_model(data):
    X = data.drop(['diagnosis'],axis=1)
    y= data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_text,y_train,y_text = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LogisticRegression()
    model.fit(X_train,y_train)


    y_pred = model.predict(X_text)
    print('Accuracy of our model: ', accuracy_score(y_text,y_pred))
    print('Classification report: \n',classification_report(y_text,y_pred))

    return model,scaler


def main():
    data = get_clean_data()

    model,scaler = create_model(data)

    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

    print("Ok")

if __name__ == '__main__':
    main()