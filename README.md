# Türkçe Restoran Yorumlarında Duygu Analizi

Yemeksepeti yorumları üzerinde metin ön işleme, puandan etiket üretimi ve klasik makine öğrenmesiyle sınıflandırma çalışması.

## İşlem akışı

1. Excel dosyasındaki yorum ve lezzet puanlarını Pandas ile okuma.
2. Küçük harfe dönüştürme, sayı/noktalama temizliği ve Türkçe durak sözcüklerini kaldırma.
3. NLTK ile tokenizasyon; JPype üzerinden Zemberek yazım önerilerini kullanma.
4. Temizlenmiş veriyi `New_yemek_sepeti.xlsx` dosyasına yazma.
5. Veriyi %80 eğitim / %20 test olarak ayırma.
6. CountVectorizer ile unigram/bigram özellikleri çıkarma.
7. Logistic Regression ve Multinomial Naive Bayes modellerini eğitme.
8. Naive Bayes için sınıflandırma raporu, doğruluk ve karışıklık matrisi üretme.

## Teknolojiler

Python, Pandas, NumPy, NLTK, scikit-learn, JPype/Zemberek, Matplotlib, Seaborn.

## Dosyalar

| Dosya | Sorumluluk |
| --- | --- |
| `united1.py` | Temizleme, etiket dönüştürme ve Zemberek ön işleme |
| `united2.py` | Özellik çıkarımı, model eğitimi ve değerlendirme |

## Çalıştırma hazırlığı

Kod, yerel Excel dosyaları, Türkçe NLTK kaynakları, Java ve Zemberek JAR dosyası gerektirir. `united1.py` içindeki JVM/JAR yolları kendi ortamınıza göre düzeltilmelidir. Ardından ön işleme çıktısı üretilip `united2.py` çalıştırılır.

Depoda sabitlenmiş bağımlılık sürümleri yoktur. Kodda eski Seaborn ve scikit-learn çağrıları bulunduğu için güncel ortamda uyumluluk düzenlemesi gerekebilir; çalıştırılabilirliği yeniden doğrulanmış bir paket olarak sunulmaz.

## Bilinen yöntem sınırları

- Etiket dönüşümünde 5'in altı 0, 7'nin üstü 1 olur; 5–7 arasındaki puanlar korunur. Bu nedenle mevcut kodun tamamı temiz bir ikili sınıflandırma hattı değildir.
- İlk düzenli ifade yalnız ASCII harflerini korur; Türkçe karakterler için ön işleme iyileştirilmelidir.
- Eğitim/test ayrımı tabakalı değildir.
- TfidfVectorizer içe aktarılmıştır; mevcut eğitim hattında kullanılan yöntem CountVectorizer'dır.
- Yeniden üretilmiş bir başarı yüzdesi paylaşılmamaktadır.

Bu çalışma, Türkçe metin işleme ve sınıflandırma akışını gösteren bir portföy projesidir.

[GitHub profili](https://github.com/Abdullatif171)
