import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Car Recommender", layout="centered")
st.title("🚗 Car Recommendation System")

@st.cache_data
 def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # Remove outliers
    def remove_outliers_iqr(df, cols):
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                df = df[(df[col] >= lb) & (df[col] <= ub)]
        return df
    df = remove_outliers_iqr(df, ["Year", "KM's driven", "Price"])
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
 def build_tfidf_matrix(df):
    df['item_profile'] = df[['Make','Model','Fuel','Transmission','Assembly','Condition']]
                          .astype(str).agg(' '.join, axis=1)
    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(df['item_profile'])
    return df, mat

@st.cache_data
 def build_onehot_matrix(df):
    features = ['Make','Model','Fuel','Transmission','Assembly','Condition']
    enc = OneHotEncoder(handle_unknown='ignore')
    mat = enc.fit_transform(df[features].astype(str))
    return enc, mat

# Sidebar controls
st.sidebar.header("Configuration")
method = st.sidebar.selectbox("Select method", [
    "TF-IDF Profile",
    "One-Hot KNN",
    "Context-Aware"
])
n_recs = st.sidebar.number_input("Number of recommendations", min_value=1, max_value=50, value=10)

uploaded = st.file_uploader("Upload OLX Cars CSV", type=["csv"])
if uploaded:
    df = load_and_clean_data(uploaded)
    st.success(f"Data loaded: {df.shape[0]} records.")

    if method in ["TF-IDF Profile", "One-Hot KNN"]:
        car_name = st.text_input("Enter exact Car Name:")
        if st.button("Get Recommendations") and car_name:
            if method == "TF-IDF Profile":
                df_mat, tfidf_mat = build_tfidf_matrix(df)
                if car_name not in df_mat['Car Name'].values:
                    st.error("Car not found.")
                else:
                    idx = df_mat.index[df_mat['Car Name'] == car_name][0]
                    nn = NearestNeighbors(n_neighbors=n_recs+1, metric='cosine')
                    nn.fit(tfidf_mat)
                    dists, inds = nn.kneighbors(tfidf_mat[idx])
                    results = pd.DataFrame([
                        {'Car Name': df_mat.iloc[i]['Car Name'], 'Score': round(1-d,3)}
                        for d,i in zip(dists[0][1:], inds[0][1:])
                    ])
                    st.dataframe(results)
            else:
                enc, onehot_mat = build_onehot_matrix(df)
                if car_name not in df['Car Name'].values:
                    st.error("Car not found.")
                else:
                    idx = df.index[df['Car Name'] == car_name][0]
                    nn = NearestNeighbors(n_neighbors=n_recs+1, metric='cosine')
                    nn.fit(onehot_mat)
                    dists, inds = nn.kneighbors(onehot_mat[idx])
                    results = pd.DataFrame([
                        {'Car Name': df.iloc[i]['Car Name'], 'Score': round(1-d,3)}
                        for d,i in zip(dists[0][1:], inds[0][1:])
                    ])
                    st.dataframe(results)

    else:  # Context-Aware
        st.subheader("User Context Preferences")
        fuel = st.selectbox("Fuel", df['Fuel'].unique().tolist())
        trans = st.selectbox("Transmission", df['Transmission'].unique().tolist())
        cond = st.selectbox("Condition", df['Condition'].unique().tolist())
        reg_city = st.selectbox("Registration City", df['Registration city'].fillna('Unknown').unique().tolist())
        seller_loc = st.selectbox("Seller Location", df['Seller Location'].fillna('Unknown').unique().tolist())
        if st.button("Get Context-Aware Recommendations"):
            ctx_cols = ['Fuel','Transmission','Condition','Registration city','Seller Location']
            user_ctx = {'Fuel': fuel, 'Transmission': trans, 'Condition': cond,
                        'Registration city': reg_city, 'Seller Location': seller_loc}
            # encode
            enc_ctx = OneHotEncoder(handle_unknown='ignore')
            item_mat = enc_ctx.fit_transform(df[ctx_cols].astype(str))
            user_vec = enc_ctx.transform(pd.DataFrame([user_ctx]).astype(str))
            sims = cosine_similarity(user_vec, item_mat).flatten()
            top_idx = sims.argsort()[::-1][:n_recs]
            results = pd.DataFrame([{
                'Car Name': df.iloc[i]['Car Name'], 'Score': round(sims[i],3),
                **{c: df.iloc[i][c] for c in ctx_cols}
            } for i in top_idx])
            st.dataframe(results)

st.markdown("---")
st.write("Built with ❤️ using Streamlit")
