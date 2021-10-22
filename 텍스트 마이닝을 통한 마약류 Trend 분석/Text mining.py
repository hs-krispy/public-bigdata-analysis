import re
import os
import apyori
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from PIL import Image

dir = "./tumblr"
plt.rc("font", family="Malgun Gothic")
kkma = Kkma()

mask_image = np.array(Image.open("C:/Users/user/Downloads/marijuana2.png"))
change = {"버닝": "버닝썬", "정국": "마약청정국", "청정국": "마약청정국", "피벌룬": "해피벌룬", "신종": "신종마약",
          "벌룬": "해피벌룬", "해피": "해피벌룬", "풍선": "해피벌룬", "질소": "아산화질소", "프로": "프로포폴", "홍정": "홍정욱",
          "펜타": "펜타닐", "아산화": "아산화질소", "우리나라": "대한민국", "다이": "다이아몬드", "아몬드": "다이아몬드",
          "다이아몬드라": "다이아몬드", "카트": "카트리지", "보디": "캄보디아", "10": "10대", "20": "20대", "30": "30대", "10대도": "10대",
          "바티칸": "바티칸 킹덤", "킹덤": "바티칸 킹덤", "다크": "다크웹", "화폐": "가상화폐", "가상": "가상화폐", "암호": "가상화폐",
          "우편": "우편 (국제우편)", "국제우편": "우편 (국제우편)", "유흥": "유흥업소", "업소": "유흥업소", "벤조디": "벤조디아제핀",
          "벤질": "벤질펜타닐", "벤질펜타": "벤질펜타닐", "코로나": "코로나19", "19": "코로나19", "차운술": "차가운술", "마초": "대마초",
          "차운": "차가운술", "텔레": "텔레그램", "헤로": "헤로인", "제식": "듀로제식", "안전텔레": "텔레그램", "안전위": "위커", "인증능": "인증"}

def WC(data, cmap):
    # 명사 추출
    data = list(map(kkma.nouns, data))

    words = []
    for word in data:
        words.extend(word)
    # 불용어 제거 & word mapping
    stopwords = pd.read_csv("./한글불용어.csv", index_col=0, encoding="CP949").Stopwords.unique()
    clear_words = [change.get(cw, cw) for cw in words if (cw not in stopwords) & (len(cw) >= 2) & (len(cw) <= 10)]

    # WordCloud 시각화
    most_common_30 = dict(Counter(clear_words).most_common(30))
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/H2GTRM.TTF", width=800, height=400, relative_scaling=0.5,
                          colormap=cmap, min_font_size=17, max_font_size=65, mask=mask_image,
                          background_color="black").generate_from_frequencies(most_common_30)

    # 불용어가 제거된 모든 단어 목록에서 빈도를 추출하고 csv로 변환
    # words_df = pd.DataFrame(clear_words, columns=["bins"])
    # words_df.bins.value_counts().to_csv(f"bins/{name} bins.csv", encoding="CP949")

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    # plt.savefig("저장경로")


def TF_IDF(data):
    stopwords = pd.read_csv("./한글불용어.csv", index_col=0, encoding="CP949").Stopwords.unique()

    # 명사 추출
    data = list(map(kkma.nouns, data))
    words_list = []
    for words in data:
        W = [change.get(word, word) for word in words if (word not in stopwords) & (len(word) >= 2) & (len(word) <= 10)]
        if len(W) > 0:
            words_list.append(W)

    # 알맞은 format으로 변환
    sentences = list(map(" ".join, words_list))

    # 최소 5개의 문장에 등장한 단어들을 대상으로 TF-IDF 추출 및 내림차순 정렬
    tfidf = TfidfVectorizer(min_df=5)
    max_val = tfidf.fit_transform(sentences).max(axis=0).toarray().ravel()
    df = pd.DataFrame(max_val, index=tfidf.get_feature_names(), columns=["tf-idf"]).sort_values(by="tf-idf",
                                                                                                ascending=False)

    return df


def association_rules(data):
    stopwords = pd.read_csv("./한글불용어.csv", index_col=0, encoding="CP949").Stopwords.unique()
    # 명사 추출
    data = list(map(kkma.nouns, data))
    words_list = []
    for words in data:
        W = [change.get(cw, cw) for cw in words if (cw not in stopwords) & (len(cw) >= 2) & (len(cw) <= 10)]
        if len(W) > 0:
            words_list.append(W)

    # 지지도, 향상도 설정
    result = pd.DataFrame(list(apyori.apriori(words_list, min_support=0.01, min_lift=1)))
    # 단어 개수 확인
    result.loc[:, "len"] = result["items"].apply(lambda x: len(x))

    G = nx.Graph()
    # 단어쌍 (2 단어씩) 추출 후 Graph에 추가
    List = list(result.loc[result.len == 2, "items"].apply(lambda x: list(x)).values)
    G.add_edges_from(List)

    # 근접 중심성 계산 & 노드 크기 설정
    cls = nx.closeness_centrality(G)
    nsize = np.array([v for v in cls.values()])
    nsize = 7000 * (nsize / max(nsize))
    fsize = 35 * (nsize / max(nsize))

    pos = nx.spring_layout(G, k=0.3)
    nx.draw_networkx(G, font_family="Malgun Gothic", node_color=list(cls.values()), cmap=plt.cm.YlOrRd,
                     node_size=nsize, pos=pos, alpha=0.85, width=1.5, edge_color="black", with_labels=False)
    # text 크기 조정
    for idx, (node, (x, y)) in enumerate(pos.items()):
        plt.text(x, y, node, fontsize=fsize[idx], color="black", ha='center', va='center')
    plt.axis("off")
    plt.tight_layout()
    plt.show()


for idx, (file, color) in enumerate(zip(os.listdir(dir), ["Blues", "autumn", "YlGn"])):
    data = open(os.path.join(dir, file), encoding="UTF-8").readlines()
    # 불필요 문장 or 단어 제거 및 개행문자 제거
    data = list(map(lambda x: re.sub("(페이지|.*@.*|<.*송고|\[.*]|.*기자|.*특파원|.*송고|.*AFP|=|.*사진.*|※.*|【.*|Play.*|▶.*"
                                     "|영상편집.*|그래픽.*|#|앵커|뉴스래빗|◇.*>|◆.*>|.*>|mbc.*|박근혜·이명박.*|명이|지난달|만명|제가|보도.*"
                                     "|.*뉴스|리포트|a.*|A.*|그는|으로|취재.*|이재명.*|.*씨는|ytn|속보.*|.*씨|윤 의원이 정책.*|10대 딸들.*"
                                     "|MBN이.*|등록.*수정|\d+(월|세|년|명|일|배|억|만|만원|kg|㎡|원|개|건|리터|시간|층|차례|여|여명|부)|■.*"
                                     "|CBS 라디오.*)", "", x).rstrip("\n"), data))
    # data = list(map(lambda x: re.sub("(반응.*|.*팔로우|\.\.\.*|@.*|문의.*|채널.*|펼치기|안녕하세요.*|\d+(지|정|팟|pill|g|시간|만|쥐|ug|종류|mg)|상담"
    #                                  "|:.*|영업시간.*|인증채널.|가격*|아이스판매처|판매처|아이스판|아이스팝|고통|아이디|#)", "", x).rstrip("/n"), data))

    WC(data, color)
    TF_IDF(data)
    association_rules(data)