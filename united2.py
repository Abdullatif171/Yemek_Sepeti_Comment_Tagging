import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns;

column = ['','yorum',"lezzet"]
data = pd.read_excel('New_yemek_sepeti.xlsx')

#Data do train and test
X_train, X_test, y_train, y_test = train_test_split(data['yorum'], data['lezzet'], test_size=0.2,random_state = 42)

vect = CountVectorizer(encoding ='iso-8859-9',max_features =8000,
                             min_df = 2,
                             ngram_range = (1,2)).fit(X_train.values.astype('U'))

X_train_vectorized = vect.transform(X_train.values.astype('U')) 

model = LogisticRegression(solver='lbfgs',random_state =0)
model.fit(X_train_vectorized, y_train)

Test = model.predict(vect.transform(X_test))

ozellik_isimleri = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('\nNegatif:\n',ozellik_isimleri[sorted_coef_index[:20]])
print('\nPozitif:\n',ozellik_isimleri[sorted_coef_index[:-21:-1]])

print('*'*20,'naive_bayes','*'*20)
from sklearn.naive_bayes import MultinomialNB

# naive bayes veri kümesine uydurma
model2 = MultinomialNB()
model2.fit(X_train_vectorized, y_train)

#Test yapma
Test2 = model2.predict(vect.transform(X_test))

#başarı oran
print(classification_report(y_test, Test2))

#hata matrisi
score = round(accuracy_score(y_test, Test2), 3)
matrix2 = confusion_matrix(y_test, Test2)
sns.heatmap(matrix2, annot=True, fmt=".0f")
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Naive Bayes Doğruluk Skoru: {0}'.format(score), size = 10)
plt.show()

