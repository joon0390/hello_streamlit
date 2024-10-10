import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 페이지 설정
st.set_page_config(page_title="KMeans 클러스터링", layout="wide")

# 타이틀
st.title("KMeans 클러스터링 🤠")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드 해주세요", type=["csv"])

# 파일이 업로드 되었는지 확인
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("데이터 미리보기")
    st.dataframe(df)

    # 사용자에게 사용할 피처 선택 옵션 제공
    columns = st.multiselect("클러스터링을 위한 컬럼을 선택해주세요", df.columns)

    if columns:
        # 사용자에게 클러스터 수 선택 옵션 제공
        n_clusters = st.slider("클러스터의 개수를 설정해주세요 (k)", min_value=2, max_value=10, value=3)

        # KMeans 클러스터링 실행
        X = df[columns].dropna()  # 결측값 제거
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # 클러스터 결과 출력
        st.subheader(f"{n_clusters}개의 클러스터로 실행한 Kmeans 클러스터링")
        st.write(f"클러스터링을 위해 선택한 컬럼: {columns}")

        # 클러스터 센터 출력
        st.subheader("클러스터 중심")
        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=columns))

        # 클러스터링 결과 시각화
        st.subheader("Cluster Visualization")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X[columns[0]], y=X[columns[1]], hue=df['Cluster'], palette="viridis", s=100, alpha=0.7)
        plt.title(f"KMeans Clustering Result with {n_clusters} Clusters")
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        st.pyplot(plt.gcf())

else:
    st.write("CSV 파일을 업로드해주세요")
