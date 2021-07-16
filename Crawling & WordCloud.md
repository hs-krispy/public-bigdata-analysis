## Crawling & WordCloud

### Crawling

- selenium을 이용한 data crawling
- 구글 검색 정보 중 일부를 추출

```python
import requests
import lxml.html
import pandas as pd
import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium import webdriver

options = webdriver.ChromeOptions()
# window size 최대화
options.add_argument("--start-maximized");
browser = webdriver.Chrome("chromedriver 경로", options=options)
url = "페이지 link"
```

- selenium을 이용해 가상 브라우저 생성

```python
browser.get(url)
browser.implicitly_wait(3)
```

- 지정된 페이지로 이동

```python
with open("기후변화가 농업에 미치는 영향.txt",  "w", encoding="UTF-8") as file:
    for i in range(2, 7):
        print(browser.find_elements_by_class_name('yuRUbf')[i].text)
        # 클릭과 뒤로가기 등의 작업을 하면 객체를 재사용 할 수 없었으므로 매 반복마다 객체를 생성
        browser.find_elements_by_class_name('yuRUbf')[i].click()
        time.sleep(1)
        # 세부 페이지에서 p 태그안의 내용들만(기사 본문) 추출 
        article = browser.find_elements_by_tag_name('p')
        # 각 내용을 지정된 file에 write
        for content in article:
            file.write(content.text)
        # 이전 페이지로 뒤로가기
        browser.back()
        time.sleep(3)
```

- 페이지의 html을 살펴본 뒤 가져오고자 하는 부분을 지정
- 원하는 부분의 데이터만 추출해서 file에 write

### WordCloud

```python
from konlpy.tag import Kkma
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

kkma = Kkma()
words = []
with open("기후변화가 농업에 미치는 영향.txt",  "r", encoding="UTF-8") as file:
    for word in file.readlines():
        words.extend(kkma.nouns(word))
    print(words)
#  '농업', '과학원', '기후', '기후변화', '변화', '태과', '연구사', '수확', '계절', '가을', '풍요', '민족', '대', '대명절', '명절', '추석', '한가위', '라', '옛말', '54', '최장', '최장기간', '기간', '장마', '폭우', '발생', '태풍', '올여름', '농작물', '침수', '피해', '범위', '2', '2만', '만', '7,633', '규모', '벼', '채소', '밭작물', '순', '지구', '지구온난화', '온난화', '기상' ...
```

- Crawling으로 얻은 data를 load하고 kkma를 이용해 명사만 추출

```python
stopwords = open("불용어.txt", "r", encoding="UTF-8")
stopwords = [sw.rstrip("\n") for sw in stopwords.readlines()] + ["기자"]
clear_words = [cw for cw in words if cw not in stopwords]
print(len(words))
print(len(clear_words))
```

- 일반적으로 한글에 적용하는 불용어를 txt 파일로 저장 후 load해서 list 형식으로 변환
- 기존 data에서 불용어 제거

```python
clear_words = dict(Counter(clear_words).most_common(100))
clear_words
# {'기후': 9,
#  '기후변화': 8,
#  '변화': 8,
#  '발생': 7,
#  '기상': 7,
#  '수': 7,
#  '농작물': 6,
#  '피해': 6,
# ...
```

- 가장 높은 빈도의 단어 100개를 dict 형태로 추출 

```python
wordcloud = WordCloud(font_path="C:/Windows/Fonts/H2HDRM.TTF", background_color="black").generate_from_frequencies(clear_words)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/125976594-6043e8dc-729e-4248-b4ba-22301aef961c.png" width=60% />

- WordCloud 생성
  - 기후변화, 농업, 이상기상, 변화, 피해, 농작물 등의 키워드가 크게 나타나는 것으로 보아 기후 변화로 인해 농업 분야의 피해가 많이 발생하고 있음을 유추할 수 있음 