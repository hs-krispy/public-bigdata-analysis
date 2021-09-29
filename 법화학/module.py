import pandas as pd
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree
from plotly.subplots import make_subplots

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
plt.rc("font", family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False


class medication:
    def __init__(self, data):

        self.df = data.copy()
        self.scaler = MinMaxScaler()

    def cleansing(self):
        # 불필요한 칼럼 제거
        self.df.drop(columns=["접수일", "연생", "채취일자", self.df.columns[-1]], inplace=True)
        # 크레아티닌 값이 20이상 300이하에 해당하고 Dose가 0이 아닌 row 추출
        self.df = self.df[(20 <= self.df["[Cr]\n(mg/dL)"]) & (self.df["[Cr]\n(mg/dL)"] <= 300) & (self.df.Dose != 0)]
        # 모체농도와 대사체농도의 ND와 < C1에 대해 0 값으로 일괄 처리
        self.df.replace(["ND", "< C1", "<C1"], 0, inplace=True)
        # 따라서 보정값들도 마찬가지로 0 값으로 처리
        self.df.loc[:, ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]] = \
            self.df[["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]].fillna(0)

        return self.df

    # def annotate(self, fig):
    #     for bar in fig.patches:
    #         fig.annotate(round(bar.get_height()),
    #                      (bar.get_x() + bar.get_width() / 2,
    #                       bar.get_height()), ha='center', va='center',
    #                      size=10, xytext=(-1, 8),
    #                      textcoords='offset points')
    #
    # def annotate_h(self, fig):
    #     for bar in fig.patches:
    #         fig.annotate(round(bar.get_width(), 2),
    #                      (bar.get_width(),
    #                       bar.get_y() + bar.get_height() / 2), ha='center', va='center',
    #                      size=10, xytext=(10, 0),
    #                      textcoords='offset points')

    # 전체 데이터를 성분별 데이터로 분리
    def split(self, data):
        drugs = {}
        for drug in data.성분명.unique():
            drugs[drug] = data[data.성분명 == drug]

        return drugs

    # quantile-cut 방식으로 각 집단별 표본의 크기를 유사하게 구간화
    def quantization(self, n, data, col):
        data.loc[:, col] = pd.qcut(data[col], n, precision=2).values

        return data

    # shapiro-wilk test
    def shapiro(self, name, data, criteria, selected_columns):
        result = []

        pt = data.pivot_table(selected_columns, index=criteria, aggfunc=stats.shapiro)
        for idx in range(len(data[criteria].unique())):
            result.append(pt.apply(lambda x: x.values[idx].pvalue))
        result = pd.concat(result, ignore_index=True, axis=1).T
        result.index = pt.index
        result.index.name = f"Shapiro Test Result\n({name})"

        return result
        # result.to_csv(f"나이구분/{name} 나이구분.csv", encoding="CP949")

    # wilcoxon rank sum test
    def wilcox(self, name, data, criteria, selected_columns):
        result = []

        for col in selected_columns:
            res = stats.ranksums(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                 data.loc[data[criteria] == data[criteria].unique()[1], col])
            result.append(res.pvalue)

        result = pd.DataFrame(result, index=selected_columns, columns=[name]).T
        result.index.name = f"Wilcoxon rank sum test Result"

        return result

    # kruskal-wallis test
    def kruskal(self, name, data, criteria, selected_columns):
        result = []
        if len(data[criteria].unique()) == 3:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col])
                result.append(res.pvalue)
        elif len(data[criteria].unique()) == 4:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col],
                                    data.loc[data[criteria] == data[criteria].unique()[3], col])
                result.append(res.pvalue)
        elif len(data[criteria].unique()) == 5:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col],
                                    data.loc[data[criteria] == data[criteria].unique()[3], col],
                                    data.loc[data[criteria] == data[criteria].unique()[4], col])
                result.append(res.pvalue)
        return pd.DataFrame(result, index=selected_columns, columns=[name]).T

    # post-hoc (bonferroni)
    def bonferroni(self, data, criteria, selected_columns):
        for col in selected_columns:
            comp = MultiComparison(data[col], data[criteria])
            result = comp.allpairtest(stats.kruskal, method="bonf")
            print(data.성분명.unique()[0], col)
            print(result[0])

    # -------------------------- boxplot --------------------------
    def boxplot(self, name, data, criteria, selected_columns):
        data[selected_columns] = self.scaler.fit_transform(data[selected_columns])
        fig = px.box(data[selected_columns + [criteria]], color=criteria,
                     category_orders={criteria: sorted(data[criteria].unique())})
        fig.update_layout(
            title={
                "text": f"{name} ({criteria}) boxplot",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "font": {"size": 20}
            },
            legend={
                "orientation": "h"
            }
        )
        fig.update_xaxes(
            tickfont={"size": 17}
        )
        plotly.offline.plot(fig)
        time.sleep(1)

    # -------------------------- heatmap --------------------------
    def heatmap(self, name, data, criteria, selected_columns, method="pearson"):

        fig, axes = plt.subplots(1, len(data[criteria].unique()), figsize=(15, 6), constrained_layout=True)

        for cr, ax in zip(data[criteria].unique(), axes.flatten()):
            sns.heatmap(data.loc[data[criteria] == cr, selected_columns].corr(method=method),
                        cmap=plt.cm.Blues, annot=True, square=True, fmt=".2f", ax=ax,
                        vmin=-1, vmax=1, annot_kws={"size": 8}, cbar_kws={"shrink": 0.3})
            ax.set_title(f"{cr} correlation", fontsize=15)
            ax.tick_params(axis="both", rotation=45, size=10)
        plt.suptitle(f"{name} ({criteria}) heatmap", fontsize=20)
        # plt.show()
        plt.savefig(f"{name} ({criteria}) heatmap.png")

    # -------------------------- pairplot --------------------------
    def pairplot(self, name, data, criteria, selected_columns):
        ax = sns.pairplot(data[selected_columns + [criteria]], hue=criteria,
                          kind="kde", plot_kws={"alpha": 0.5, "common_norm": False}, diag_kind="kde",
                          palette=sns.color_palette(n_colors=len(data[criteria].unique()), palette="tab10"))
        ax._legend.remove()
        handles = ax._legend_data.values()
        labels = ax._legend_data.keys()
        plt.suptitle(f"{name} ({criteria}) pairplot", fontsize=20)
        ax.fig.legend(title=criteria, handles=handles, labels=labels, loc="upper right",
                      title_fontsize=12, fontsize=12, ncol=3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{name} ({criteria}) pairplot.png")

    # -------------------------- violinplot --------------------------
    def violinplot(self, name, data, criteria, selected_columns):
        if len(selected_columns) == 1:
            ax = sns.violinplot(x=selected_columns[0], y=criteria, data=data, inner="quartile",
                                order=sorted(data[criteria].unique()),
                                orient="h")
            Mean = data.pivot_table(selected_columns[0], index=criteria, aggfunc=["mean"]).values
            Median = data.pivot_table(selected_columns[0], index=criteria, aggfunc=["median"]).values
            ax.set_ylabel(f"{criteria}", fontsize=12, labelpad=20, rotation=0)
            ax.set_xlabel(f"{selected_columns[0]}", fontsize=12)
            ax.scatter(Mean, range(len(Mean)), c="b", alpha=0.5, label="mean")
            ax.scatter(Median, range(len(Median)), c="r", alpha=0.5, label="median")
            ax.legend(loc="best")
            ax.set_title(f"{name} ({criteria}) violinplot", fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{name} ({criteria}) violinplot.png")
        else:
            fig, axes = plt.subplots(len(selected_columns), 1, figsize=(10, 13), constrained_layout=True)

            for sc, ax in zip(selected_columns, axes.flatten()):
                Mean = data.pivot_table(sc, index=criteria, aggfunc=["mean"])
                Median = data.pivot_table(sc, index=criteria, aggfunc=["median"])
                sns.violinplot(x=sc, y=criteria, data=data, ax=ax, inner="quartile", order=Mean.index,
                               orient="h")
                ax.set_ylabel(f"{criteria}", fontsize=12, labelpad=20, rotation=0)
                ax.set_xlabel(f"{sc}", fontsize=12)
                ax.scatter(Mean.values, range(len(Mean.values)), c="b", alpha=0.5, label="mean")
                ax.scatter(Median.values, range(len(Median.values)), c="r", alpha=0.5, label="median")
                ax.legend(loc="best")
                plt.suptitle(f"{name} ({criteria}) violinplot", fontsize=15)
            # plt.show()
            plt.savefig(f"{name} ({criteria}) violinplot.png")

    # -------------------------- Q-Q plot --------------------------
    def qqplot(self, name, data, criteria, selected_columns):
        if len(selected_columns) == 1:
            fig, axes = plt.subplots(1, len(data[criteria].unique()), figsize=(15, 12),
                                     constrained_layout=True)
            for cr, ax in zip(data[criteria].unique(), axes):
                stats.probplot(data.loc[data[criteria] == cr, selected_columns[0]], plot=ax)
                ax.set_title(f"{cr} {selected_columns[0]}", fontsize=12)
                plt.suptitle(f"{name} ({criteria}) qqplot", fontsize=15)
            plt.savefig(f"{name} ({criteria}) qqplot.png")
        else:
            for sc in selected_columns:
                fig, axes = plt.subplots(len(data[criteria].unique()), 1, figsize=(5, 10),
                                         constrained_layout=True)
                for cr, ax in zip(data[criteria].unique(), axes):
                    stats.probplot(data.loc[data[criteria] == cr, sc], plot=ax)
                    ax.set_title(f"{cr} {sc}", fontsize=12)
                plt.suptitle(f"{name} ({criteria}) qqplot", fontsize=15)
                # plt.show()
                COL = sc.split("\n")[0]
                plt.savefig(f"{name} ({criteria}) ({COL}) qqplot.png")

    # -------------------------- PCA --------------------------
    def pca(self, data, n):
        pca = PCA(n_components=n, random_state=42)
        print(f"Principal Components explained variance ratio\n{pca.explained_variance_ratio_}")
        decomp_data = pca.fit_transform(data)

        return decomp_data


df = pd.read_excel("C:/Users/user/Desktop/practice/210906_복약데이터_통계.xlsx")

obj = medication(df)
cleansed_data = obj.cleansing()
cleansed_data.loc[:, ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]] = cleansed_data.loc[:, ["모체농도보정\n(ng/mg)",
                                                                                      "대사체농도보정\n(ng/mg)"]].div(
    cleansed_data.Dose, axis=0)
cleansed_data.loc[:, "Ratio2"] = cleansed_data["모체농도보정\n(ng/mg)"] / cleansed_data["대사체농도보정\n(ng/mg)"]
cleansed_data.loc[:, "Diff"] = cleansed_data["모체농도보정\n(ng/mg)"] - cleansed_data["대사체농도보정\n(ng/mg)"]

selected_columns = ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Ratio"]

# scaler = StandardScaler()
scaler = PowerTransformer(method="yeo-johnson")
for (key, value), k in zip(obj.split(cleansed_data).items(), [6, 7, 4, 7]):
    print(key)
    value.dropna(inplace=True)
    an_scaled_data = scaler.fit_transform(value[selected_columns + ["Ratio2"]])
    forest = IsolationForest(random_state=42)
    # 이상치 비율 조정
    if value.결과.value_counts()[value.결과.value_counts().index == "확인필요"].values / value.shape[0] > 0.1:
        ratio = (value.결과.value_counts()[value.결과.value_counts().index == "확인필요"].values / value.shape[0])[0]
        forest.contamination = ratio
    pred = forest.fit_predict(an_scaled_data)
    value.loc[:, "anomaly"] = pred
    # print(value.anomaly.value_counts())

    # whole data clustering
    scaled_data = scaler.fit_transform(value.loc[:, selected_columns + ["Ratio2", "Diff"]])
    cluster = KMeans(n_clusters=k, init="k-means++", random_state=42)
    label = cluster.fit_predict(scaled_data)
    value.loc[:, "cluster"] = label
    # print(value.cluster.value_counts())
    # print(silhouette_score(scaled_data, label))

    # only anormaly data clustering

    # scaled_data = scaler.fit_transform(value.loc[value.anomaly == -1, selected_columns + ["Ratio2", "Diff"]])
    # Anm = value.loc[value.anomaly == -1]
    # Anm.loc[:, "cluster"] = label
    # print(Anm.cluster.value_counts())

    # dimension reduction

    decomp_data = obj.pca(scaled_data, 3)
    # tsne = TSNE(n_components=3, random_state=42)
    # decomp_data = tsne.fit_transform(scaled_data)

    # plot 2 or 3d scatter

    # fig = px.scatter(x=scaled_data[:, 0], y=scaled_data[:, 1], color=label)
    # fig = px.scatter_3d(x=decomp_data[:, 0], y=decomp_data[:, 1], z=decomp_data[:, 2], color=label)
    # fig.show()

    cleansed_data.loc[cleansed_data.index.isin(value.index), "cluster"] = value.cluster.values

    # 통계적 검증이 필요할 때에 cluster에 속한 값이 1개인 클러스터는 제외

    # cleansed_data.loc[cleansed_data.index.isin(value.index), "anomaly"] = value.anomaly.values
    # try:
    #     idx = list(value.cluster.value_counts().values).index(1)
    #     drop_cluster = value.cluster.value_counts().index[idx]
    #     value = value[~value.cluster.isin([drop_cluster])]
    # except:
    #     pass

    # value_counts = value.cluster.value_counts()
    # print(value_counts)
    # print(value.pivot_table(["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Dose"], index="cluster", aggfunc="describe").T)
    # value.loc[:, "cluster"] = value.cluster.apply(lambda x: str(x) + f" ({value_counts[value_counts.index == x].values[0]})")
    # fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    # for col, ax in zip(["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Dose"], axes.flatten()):
    #     pt = value.pivot_table(col, index="cluster", aggfunc="describe").T
    #     pt.plot.bar(ax=ax, rot=0)
    #     ax.set_xlabel(col, fontsize=13)
    # plt.suptitle(key, fontsize=20)
    # plt.savefig(f"cluster/cluster's value ({key}).png")

cleansed_data.to_csv("add_cluster data (Dose scaling, clustering, ratio2, diff, powertransform).csv", encoding="CP949",
                     index=False)

# 통계적 검증 & visualize

# s_result = obj.shapiro(key, value, "cluster", selected_columns)
# s_result.to_csv(f"shapiro result ({key}).csv", encoding="CP949")
# k_result = obj.kruskal(key, value, "cluster", selected_columns)
# k_result.to_csv(f"cluster/kruskal result ({key}).csv", encoding="CP949")
# obj.bonferroni(value, "cluster", selected_columns)
# obj.boxplot(key, value, "cluster", selected_columns)
# obj.heatmap(key, value, "cluster", selected_columns, "spearman")
# obj.pairplot(key, value, "cluster", selected_columns)
# obj.violinplot(key, value, "cluster", selected_columns)
# obj.qqplot(key, value, "cluster", selected_columns)


# inertias = []
# silhouettes = []
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# for k, ax in zip(range(2, 8), axes.flatten()):
#     #     fig = plt.figure(figsize=(10, 8))
#     #     ax = fig.add_subplot(111, projection='3d')
#     cluster = KMeans(n_clusters=k, init="k-means++", random_state=42)
#     cluster.fit(scaled_data)
#
#     inertias.append(cluster.inertia_)
#     silhouettes.append(silhouette_score(scaled_data, cluster.labels_))
#
#     silhouettes_coefficients = silhouette_samples(scaled_data, cluster.labels_)
#     padding = len(scaled_data) // 5
#     pos = padding
#
#     ticks = []
#
#     for i in range(k):
#         coeffs = silhouettes_coefficients[cluster.labels_ == i]
#         coeffs.sort()
#
#         color = plt.cm.Spectral(i / k)
#         ax.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor="k", alpha=0.7)
#         ticks.append(pos + len(coeffs) // 2)
#         pos += len(coeffs) + padding
#     ax.yaxis.set_major_locator(FixedLocator(ticks))
#     ax.yaxis.set_major_formatter(FixedFormatter(range(k)))
#     ax.axvline(x=silhouette_score(scaled_data, cluster.labels_), color="red", linestyle="--")
#     ax.set_title(f"k={k}", fontsize=15)
# fig.text(0.1, 0.5, "Cluster", fontsize=17, ha="center", va="center", rotation=90)
# fig.text(0.5, 0.05, "Silhouette score", fontsize=17, ha="center", va="center")
# plt.suptitle(f"Silhouette diagram ({key})", fontsize=20)
# plt.savefig(f"anomaly cluster/silhouette diagram (dose scaling, ratio2, diff, powertransform, anomaly X) ({key}).png")


# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig1 = px.line(x=range(2, 8), y=inertias)
# fig2 = px.line(x=range(2, 8), y=silhouettes)
# fig2.update_traces(yaxis="y2")
# fig.add_traces(fig1.data + fig2.data)
# fig.for_each_trace(lambda t: t.update(line={"color": t.marker.color}))
# fig.update_layout(
#             title={
#                 "text": f"Kmeans++ Inertia & Silhouette score ({key})",
#                 "xanchor": "center",
#                 "yanchor": "top",
#                 "x": 0.5,
#                 "font": {"size": 25}
#             },
#             legend={
#                 "orientation": "h"
#             }
#         )
# fig.update_xaxes(
#     title={
#         "text": "K",
#         "font": {"size": 20}
#     },
#     tickfont={
#         "size": 15
#     }
# )
# fig.update_yaxes(
#     title={
#         "text": "Inertia",
#         "font": {"size": 20}
#     },
#     tickfont={
#         "size": 15
#     },
#     secondary_y=False
# )
# fig.update_yaxes(
#     title={
#         "text": "Silhouette score",
#         "font": {"size": 20}
#     },
#     tickfont={
#         "size": 15
#     },
#     secondary_y=True
# )
# fig.show()
