## Crawling & WordCloud

#### WordCloud

```python
from konlpy.tag import Kkma
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
kkma = Kkma()
plt.rc("font", family="Malgun Gothic")
```

```python
with open("C:/Users/0864h/Desktop/기후변화에 따른 농업 피해.txt", "r", encoding="UTF-8") as text:
    words = []
    for t in text:
        words.extend(kkma.nouns(t))
    print(len(words))
```

- wordcloud를 생성하고자 하는 대상에 대해 kkma를 이용해 명사만 추출

```python
stopwords = open("C:/Users/0864h/Desktop/불용어.txt", "r", encoding="UTF-8")
stopwords = [sw.rstrip("\n") for sw in stopwords.readlines()]
clear_words = [cw for cw in words if cw not in stopwords]
print(len(clear_words))
```

- 일반적으로 한글에 적용하는 불용어를 txt 파일로 저장 후 load해서 list 형식으로 변환
- 불용어 제거

```python
data = dict(Counter(clear_words).most_common(100))
data
# {'기후': 25,
#  '피해': 24,
#  '농민': 21,
#  '이상기후': 15,
#  '수': 14,
#  '농작물': 14,
#  '재해': 14,
#  '정부': 13,
#  '농업': 12,
#  '보험': 10,
# ...
```

- 가장 높은 빈도의 단어 100개를 dict 형태로 추출 

```python
wordcloud = WordCloud(font_path="C:/Windows/Fonts/H2GPRM.TTF", relative_scaling=0.2, 
                      background_color="black").generate_from_frequencies(data)
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

- WordCloud 생성
  - 농민, 이상기후, 기후변화, 피해, 재해 등의 키워드가 크게 나타나는 것으로 보아 기후 변화로 인한 농민의 피해가 많이 발생함을 유추할 수 있음 

<img src="https://user-images.githubusercontent.com/58063806/125962578-7fb6cf4e-ac86-4f45-9c8f-2af53d513a9d.png" width=50% />