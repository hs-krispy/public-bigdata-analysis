import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import apyori
import networkx as nx
from konlpy.tag import Kkma
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain

dir = "./article"
plt.rc("font", family="Malgun Gothic")
kkma = Kkma()

change = {"버닝": "버닝썬", "정국": "마약청정국", "청정국": "마약청정국", "피벌룬": "해피벌룬", "신종": "신종마약",
          "벌룬": "해피벌룬", "해피": "해피벌룬", "풍선": "해피벌룬", "질소": "아산화질소", "프로": "프로포폴", "홍정": "홍정욱",
          "펜타": "펜타닐", "아산화": "아산화질소", "우리나라": "대한민국", "다이": "다이아몬드", "아몬드": "다이아몬드",
          "다이아몬드라": "다이아몬드", "카트": "카트리지", "보디": "캄보디아", "10": "10대", "20": "20대", "30": "30대", "10대도": "10대",
          "바티칸": "바티칸 킹덤", "킹덤": "바티칸 킹덤", "다크": "다크웹", "화폐": "가상화폐", "가상": "가상화폐", "암호": "가상화폐",
          "우편": "우편 (국제우편)", "국제우편": "우편 (국제우편)", "유흥": "유흥업소", "업소": "유흥업소", "벤조디": "벤조디아제핀",
          "벤질": "벤질펜타닐", "벤질펜타": "벤질펜타닐", "코로나": "코로나19", "19": "코로나19", "차운술": "차가운술", "마초": "대마초",
          "차운": "차가운술", "텔레": "텔레그램", "헤로": "헤로인", "제식": "듀로제식", "안전텔레": "텔레그램", "안전위": "위커", "인증능": "인증"}


class Text_mining:
    def __init__(self, data, mapping_table, stopwords):
        self.data = data.copy()
        self.mapping_table = mapping_table.copy()
        self.stopwords = stopwords.copy()
        self.clear_words = []

    # 분석 대상이 되는 텍스트 파일의 내용과 목적에 맞게 수정 필요
    def preprocessing(self):

        # 불필요 문장 or 단어 제거 및 개행문자 제거 (데이터 정제)
        cleansed_data = list(
            map(lambda x: re.sub("(페이지|.*@.*|<.*송고|\[.*]|.*기자|.*특파원|.*송고|.*AFP|=|.*사진.*|※.*|【.*|Play.*|▶.*"
                                 "|영상편집.*|그래픽.*|#|앵커|뉴스래빗|◇.*>|◆.*>|.*>|mbc.*|박근혜·이명박.*|명이|지난달|만명|제가|보도.*"
                                 "|.*뉴스|리포트|a.*|A.*|그는|으로|취재.*|이재명.*|.*씨는|ytn|속보.*|.*씨|윤 의원이 정책.*|10대 딸들.*"
                                 "|MBN이.*|등록.*수정|\d+(월|세|년|명|일|배|억|만|만원|kg|㎡|원|개|건|리터|시간|층|차례|여|여명|부)|■.*"
                                 "|CBS 라디오.*)", "", x).rstrip("\n"), self.data))
        # 정제된 데이터를 바탕으로 명사 추출
        Nouns_list = list(map(kkma.nouns, cleansed_data))

        # 각 명사집단 중 불용어 제거, 단어 길이제한 & word mapping 과정을 거쳐 1개 이상의 단어가 남아있는 경우만 input 으로
        for words in Nouns_list:
            Nouns = [self.mapping_table.get(word, word) for word in words if
                     (word not in self.stopwords) & (len(word) >= 2) & (len(word) <= 10)]
            if len(Nouns) > 0:
                self.clear_words.append(Nouns)

    # 필요에 따라 Wordcloud shape 이미지 file 지정가능
    def WC(self, mask=None):
        # 불용어 제거, 단어 길이제한 & word mapping
        input_data = []
        [input_data.extend(words) for words in self.clear_words]

        # 전처리가 끝난 단어 목록에서 빈도를 추출하고 csv 파일로 변환
        # words_df = pd.DataFrame(input_data, columns=["bins"])
        # words_df.bins.value_counts().to_csv(f"bins/{name} bins.csv", encoding="CP949")

        # WordCloud 시각화
        # 파라미터들은 필요에 따라 조정 필요
        most_common_30 = dict(Counter(input_data).most_common(30))

        wordcloud = WordCloud(font_path="C:/Windows/Fonts/H2GTRM.TTF", width=800, height=400, relative_scaling=0.5,
                              colormap=cmap, min_font_size=17, max_font_size=65, mask=mask,
                              background_color="black").generate_from_frequencies(most_common_30)

        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud)
        plt.axis("off")
        # 결과 출력
        plt.show()
        # 결과 저장
        # plt.savefig("저장경로")

    def TF_IDF(self):

        # 알맞은 format으로 변환
        input_data = list(map(" ".join, self.clear_words))
        print(input_data)
        # 최소 5개의 문장에 등장한 단어들을 대상으로 각 문장에서 가장 높은 TF-IDF 값을 갖는 단어와 그 값을 추출
        tfidf = TfidfVectorizer(min_df=5)
        max_val = tfidf.fit_transform(input_data).max(axis=0).toarray().ravel()
        df = pd.DataFrame(max_val, index=tfidf.get_feature_names(), columns=["tf-idf"]).sort_values(by="tf-idf",
                                                                                                    ascending=False)
        # 결과 저장
        # 한글의 경우 encoding CP949, 영어는 UTF-8
        # df.to_csv("저장 경로", encoding="CP949")

        return df

    def association_rules(self):

        input_data = self.clear_words

        # 지지도 (support), 향상도 (lift) 설정
        result = pd.DataFrame(list(apyori.apriori(input_data, min_support=0.01, min_lift=1)))
        # 각각의 단어들의 개수 확인
        result.loc[:, "len"] = result["items"].apply(lambda x: len(x))

        G = nx.Graph()
        # 1 대 1 대응(A <-> B) 경우만 추출 후 Graph에 추가
        List = list(result.loc[result.len == 2, "items"].apply(lambda x: list(x)).values)
        G.add_edges_from(List)

        # 그래프의 모듈성을 최적화하는 louvain algorithm, 각 노드(단어)별 색상(군집)을 결정
        bb = community_louvain.best_partition(G)
        colors = [bb[word] for word in G.nodes]

        # 근접 중심성 계산 & 노드 크기 설정
        cls = nx.closeness_centrality(G)
        cls_list = np.array([v for v in cls.values()])
        node_size = 7000 * (cls_list - min(cls_list)) / (max(cls_list) - min(cls_list))
        font_size = 35 * (node_size / max(node_size))

        # 그래프 layout 설정 및 시각화, 파라미터는 필요에 따라 조정 필요
        pos = nx.spring_layout(G, k=0.6, seed=42)
        nx.draw_networkx(G, font_family="Malgun Gothic", node_color=colors, cmap=plt.cm.Set3, node_shape='h',
                         node_size=node_size, pos=pos, alpha=0.85, width=3, edge_color="black", with_labels=False)
        for idx, (node, (x, y)) in enumerate(pos.items()):
            plt.text(x, y, node, fontsize=font_size[idx], color="black", ha='center', va='center')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        # 결과 저장
        # plt.savefig("저장경로")


stopwords = pd.read_csv("./한글불용어.csv", index_col=0, encoding="CP949").Stopwords.unique()
for idx, (file, color) in enumerate(zip(os.listdir(dir), ["Blues", "autumn", "YlGn"])):
    data = open(os.path.join(dir, file), encoding="UTF-8").readlines()
    # Wordcloud shape 이미지
    mask_image = np.array(Image.open("파일 경로"))

    # 객체 생성
    tm = Text_mining(data, change, stopwords)
    tm.preprocessing()
    tm.WC(mask_image)
    tm.TF_IDF()
    tm.association_rules()
