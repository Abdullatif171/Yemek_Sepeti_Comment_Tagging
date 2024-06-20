import pandas as pd
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import jpype
from os.path import join
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
import seaborn as sns;
column = ['yorum',"lezzet"]
data = pd.read_excel('yemeklerin_sepeti1.xlsx')
data.head()
data['lezzet'] = data['lezzet'].astype(int)

data.loc[data['lezzet']<5, 'lezzet'] = 0
data.loc[data['lezzet'] >7, 'lezzet'] = 1


x=data['lezzet'].value_counts()
x=x.sort_index()
plt.figure(figsize=(4,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Yapılan yorum Sayısı")
plt.ylabel('yorum Sayısı')
plt.xlabel('yorum Sınıfı')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show();

################### Tüm karakterler küçük harfe dönüştürme ########################
def lowesterWord(yorum):
    
    yorum= re.sub("[^a-zA-Z]", " ", str(yorum))
    yorum = yorum.lower()
    return yorum

data['yorum'] = data['yorum'].apply(lowesterWord)

########################### Sayılar kaldırma #####################################
def deletcount(yorum):
     count=('0','1','2','3','4','5','6','7','8','9')
     yorum = ''.join([c for c in yorum if c not in count])
     return yorum
 
data['yorum'] = data['yorum'].apply(deletcount)

################### Noktalama işaretlerini kaldırma ############################
def deletdat(yorum):
    
     yorum = ''.join([c for c in yorum if c not in punctuation])
     return yorum

data['yorum'] = data['yorum'].apply(deletdat)

################ Şapkalı karakterleri eşleniği ile değiştirme ####################
def changechar(yorum):   
  # degistir listesindeki ilk öğeyi ikincisi ile değiştiriyoruz. 
  #Yani şapkalı harfleri normale çeviriyoruz.
  replace_letter = [('â', 'a'), ('ê', 'e'), ('î', 'i'), ('ô', 'o'), ('û', 'u')]
  for tpl in replace_letter:
     yorum = yorum.replace(tpl[0], tpl[1])
  return yorum

data['yorum'] = data['yorum'].apply(changechar)

############# Belirtilen alfabede olmayan tüm karakterleri kaldırma ##################
def cleanalfabe(yorum):
    alfabe ='ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZabcçdefgğhıijklmnoöprsştuüvyz '
    cleaned_text = ''
    for char in yorum:
        if char in alfabe:
            cleaned_text = cleaned_text + char

    return cleaned_text

data['yorum'] = data['yorum'].apply(cleanalfabe)

############################### stopword kaldırma ##################################
def deletstopWords(yorum):
   
 stopWords = set(stopwords.words('turkish'))
 words = word_tokenize(yorum)
 wordsFiltered = []
 
 for w in words:
     if w not in stopWords:
        wordsFiltered.append(w)
        yorum='  '.join(wordsFiltered)       
 return yorum

data['yorum'] = data['yorum'].apply(deletstopWords)

########################### tekrarlanan kelimeler kaldırma ########################
def deletTryWords(yorum):
 word_data =yorum 
 nltk_tokens = nltk.word_tokenize(word_data)

 ordered_tokens = set()
 result = []
 for word in nltk_tokens:
    if word not in ordered_tokens:
        ordered_tokens.add(word)
        result.append(word)
        yorum=' '.join(result)   
 return yorum

data['yorum'] = data['yorum'].apply(deletTryWords)

data.head()

########################## Zemberek Kütüphanesi İle Yazım Düzeltme #################

if __name__ == '__main__':

    ZEMBEREK_PATH = r'../input/zemberek/zemberek-full.jar'
    jpype.startJVM(jpype.getDefaultJVMPath(), 'C:\Program Files\Java\jdk-15.0.2\bin\server\jvm.dll' % (ZEMBEREK_PATH), '-ea')
    

    TurkishMorphology=jpype.JClass('zemberek.morphology.TurkishMorphology')
    TurkishSpellChecker=jpype.JClass('zemberek.normalization.TurkishSpellChecker')
    TurkishSentenceNormalizer = jpype.JClass('zemberek.normalization.TurkishSentenceNormalizer')

    morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()

    spell_checker: TurkishSpellChecker = TurkishSpellChecker(morphology)

def kelimeduzeltme(yorum):
 words = word_tokenize(yorum)
 

 for i, word in enumerate(words):
        if spell_checker.suggestForWord(JString(word)):
            if not spell_checker.check(JString(word)):
                words[i] = str(spell_checker.suggestForWord(JString(word))[0])
              

 yorum=' '.join(words) 
 

 return yorum

data['yorum'] = data["yorum"].apply(kelimeduzeltme)
print(data.head(5))
jpype.shutdownJVM()

data.to_excel('New_yemek_sepeti.xlsx')
