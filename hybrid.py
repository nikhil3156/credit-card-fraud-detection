# app.py (updated, robust X_iso handling)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Page config ----------------
st.set_page_config(page_title="Hybrid Fraud Detection", layout="wide", page_icon="🛡️")
# ---------------- Styling --------------------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#f6fbff 0%, #ffffff 100%); }
      .title { font-weight:800; font-size:30px; color:#0b3d91; }
      .subtitle { color:#0b3d91; font-weight:700; }
      .kpi { border-radius:12px; padding:16px; background: linear-gradient(180deg,#ffffff 0%, #f7fbff 100%); box-shadow: 0 6px 20px rgba(12,38,63,0.06); }
      .small { color:#566274; }
      .highlight { background: linear-gradient(90deg,#fffaf0,#fff1e6); padding:10px;border-radius:10px; border: 1px solid #ffd9a6; }
      .bold { font-weight:700; }
      .footer { color:#868e96; font-size:13px; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------- Header ---------------------
header_col1, header_col2 = st.columns([1, 7])
with header_col1:
    dev_image_path = "/mnt/data/77880902-87ea-4df3-bbaf-4204283d6896.png"
    if os.path.exists(dev_image_path):
        st.image(dev_image_path, width=84)
    else:
        st.write("")

with header_col2:
    st.markdown('<div class="title">Hybrid Fraud Detection — Supervised + Unsupervised</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Upload unlabeled bank transactions and (optionally) your pre-trained models to get final risk scores and insights.</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sidebar --------------------
st.sidebar.header("1) Upload data & models")
csv_file = st.sidebar.file_uploader("financial fraud.csv", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo data (fast)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Add your models** (optional)")

iso_file = st.sidebar.file_uploader("isoo_model.pkl", type=["pkl","joblib"])
st.sidebar.markdown('<div class="highlight"><b>Tip:</b> Save your iso model using:<br><code>joblib.dump({\"iso\":iso, \"scaler\":scaler, \"features\":features}, "iso_model.pkl")</code></div>', unsafe_allow_html=True)
st.sidebar.markdown("OR place the file at: <code>models/iso_model.pkl</code> and the app will try to load it automatically.")
st.sidebar.markdown("---")
credit_file = st.sidebar.file_uploader("credit_card_model.pkl", type=["pkl","joblib"])

contamination = st.sidebar.slider("IsolationForest contamination (if not using saved iso)", 0.001, 0.1, 0.02, step=0.001)
sup_weight = st.sidebar.slider("Supervised weight in final score", 0.0, 1.0, 0.7, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ — Hybrid approach (supervised + anomaly).")

# ---------------- Load or demo data --------------------
@st.cache_data
def load_demo(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "step": rng.randint(1, 10000, size=n),
        "amount": np.abs(rng.normal(100, 300, size=n)),
        "oldbalanceOrg": np.abs(rng.normal(10000, 30000, size=n)),
        "newbalanceOrig": np.abs(rng.normal(9000, 30000, size=n)),
        "oldbalanceDest": np.abs(rng.normal(5000, 20000, size=n)),
        "newbalanceDest": np.abs(rng.normal(4000, 20000, size=n)),
        "isFraud": rng.choice([0,0,0,1], size=n, p=[0.97,0.01,0.01,0.01])
    })
    for i in rng.choice(n, 10, replace=False):
        df.loc[i, "amount"] *= 200
        df.loc[i, "isFraud"] = 1
    return df

if use_demo:
    df = load_demo()
    st.success("Loaded demo dataset.")
elif csv_file:
    df = pd.read_csv(csv_file, low_memory=False)
    st.success("CSV loaded.")
else:
    st.warning("Please upload a transactions CSV or enable 'Use demo data' in the sidebar.")
    st.stop()

# ---------------- Basic preview --------------------
st.markdown("### Dataset preview")
st.dataframe(df.head(6))

# ---------------- Feature mapping --------------------
st.markdown("### <span class='subtitle'>Feature mapping & preprocessing</span>", unsafe_allow_html=True)
def find_col(df, names):
    names = [n.lower() for n in names]
    for c in df.columns:
        for n in names:
            if n in c.lower():
                return c
    return None

amount_col = find_col(df, ["amount","amt"])
step_col = find_col(df, ["step","time","timestamp"])
old_org_col = find_col(df, ["oldbalanceorg","oldbalance_org","oldbalance"])
old_dest_col = find_col(df, ["oldbalancedest","oldbalancedest","oldbalance_dest","oldbalancedest"])

if amount_col is None:
    st.error("No amount column detected. Please ensure your CSV has a column named like 'amount'.")
    st.stop()

df['amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
df['step'] = pd.to_numeric(df[step_col], errors='coerce').fillna(0).astype(int) if step_col else 0
df['oldbalanceOrg'] = pd.to_numeric(df[old_org_col], errors='coerce').fillna(0.0) if old_org_col else 0.0
df['oldbalanceDest'] = pd.to_numeric(df[old_dest_col], errors='coerce').fillna(0.0) if old_dest_col else 0.0

df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)
name_dest_col = find_col(df, ["namedest","name_dest","merchant","dest"])
if name_dest_col:
    df['dest_is_customer'] = df[name_dest_col].astype(str).str.startswith('C').astype(int)
else:
    df['dest_is_customer'] = 0

features = ['amount','step','amount_ratio','oldbalanceOrg','oldbalanceDest','dest_is_customer']
st.write("Using features:", features)

# ---------------- Load iso model / prepare X_iso robustly ----------------
st.markdown("### <span class='subtitle'>Unsupervised model (IsolationForest)</span>", unsafe_allow_html=True)

iso = None
saved_scaler = None
saved_features = None
X_iso = None           # ensure defined
scaler_used = None
loaded_iso_source = None

# 1) Try uploader
if iso_file:
    try:
        iso_obj = joblib.load(iso_file)
        if isinstance(iso_obj, dict) and {'iso','scaler','features'}.issubset(set(iso_obj.keys())):
            iso = iso_obj['iso']
            saved_scaler = iso_obj['scaler']
            saved_features = iso_obj['features']
            loaded_iso_source = "uploaded_dict"
            st.success("Loaded iso_model.pkl (dict) from uploader.")
        else:
            iso = iso_obj
            loaded_iso_source = "uploaded_raw_model"
            st.warning("Uploaded iso file loaded as model object (no scaler/features found).")
    except Exception as e:
        iso = None
        st.error("Failed to load uploaded iso file: " + str(e))

# 2) Try local models/iso_model.pkl if uploader not used or failed
if iso is None and os.path.exists("models/iso_model.pkl"):
    try:
        iso_obj = joblib.load("models/iso_model.pkl")
        if isinstance(iso_obj, dict) and {'iso','scaler','features'}.issubset(set(iso_obj.keys())):
            iso = iso_obj['iso']
            saved_scaler = iso_obj['scaler']
            saved_features = iso_obj['features']
            loaded_iso_source = "local_dict"
            st.success("Loaded models/iso_model.pkl from disk (dict).")
        else:
            iso = iso_obj
            loaded_iso_source = "local_raw"
            st.warning("Loaded models/iso_model.pkl but it's a raw model (no scaler/features).")
    except Exception as e:
        st.error("Could not load models/iso_model.pkl: " + str(e))
        iso = None

# 3) Decide features to use
if saved_features:
    feat_for_iso = saved_features
else:
    feat_for_iso = features  # fallback to detected features

# 4) If we have a saved scaler, try to use it to transform the data
if saved_scaler is not None:
    try:
        X_iso = saved_scaler.transform(df[feat_for_iso].fillna(0))
        scaler_used = "saved_scaler"
    except Exception as e:
        st.warning("Saved scaler could not transform current data (column mismatch). Will refit scaler locally. Error: " + str(e))
        saved_scaler = None
        X_iso = None
        scaler_used = None

# 5) If no saved scaler or transform failed, fit a local scaler on feat_for_iso
if X_iso is None:
    try:
        local_scaler = StandardScaler()
        X_iso = local_scaler.fit_transform(df[feat_for_iso].fillna(0).astype(float))
        saved_scaler = local_scaler
        scaler_used = scaler_used or "local_fit"
    except Exception as e:
        st.error("Failed to build feature matrix for IsolationForest. Check features and data types. Error: " + str(e))
        st.stop()

# 6) If no iso model loaded, train a fresh IsolationForest (fallback)
if iso is None:
    st.info("No pre-trained IsolationForest found. Training a new IsolationForest with current data (this may take a moment).")
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X_iso)
    loaded_iso_source = "trained_locally"
    st.success("Trained a new IsolationForest locally.")

# 7) At this point, iso and X_iso are guaranteed to exist. Compute anomaly scores below.
st.write(f"Anomaly model source: **{loaded_iso_source or 'unknown'}**, scaler used: **{scaler_used or 'unknown'}**")
# If iso is a dict, try to extract the actual estimator and scaler/features
if isinstance(iso, dict):
    # case 1: expected dict with keys 'iso','scaler','features'
    if 'iso' in iso and hasattr(iso['iso'], 'decision_function'):
        saved_iso = iso['iso']
        # update iso, and if scaler/features exist, keep them
        iso = saved_iso
        if 'scaler' in iso:
            saved_scaler = iso['scaler']
        if 'features' in iso:
            saved_features = iso['features']
    else:
        # case 2: unknown dict shape: find the first value that looks like an sklearn estimator
        found = False
        for k, v in iso.items():
            if hasattr(v, 'decision_function'):
                iso = v
                found = True
                st.warning(f"Found estimator in iso dict under key: {k}")
                break
        if not found:
            st.error("Loaded iso object is a dict but no estimator with 'decision_function' found. "
                     "Please re-save iso_model.pkl as: joblib.dump({'iso':iso, 'scaler':scaler, 'features':features}, 'iso_model.pkl')")
            st.stop()

# Final safety check
if not hasattr(iso, 'decision_function'):
    st.error("The loaded iso object does not have 'decision_function' method. Ensure you uploaded a proper IsolationForest model or a dict containing it.")
    st.stop()

# Now iso is guaranteed to be an estimator (IsolationForest). Proceed to compute scores.
# Compute anomaly scores
raw_scores = iso.decision_function(X_iso)  # higher = more normal
anomaly_prob = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
df['anomaly_prob'] = anomaly_prob
df['anomaly_flag'] = (iso.predict(X_iso) == -1).astype(int)

# ---------------- Supervised model (optional) -------------------
st.markdown("### <span class='subtitle'>Supervised score (credit model or fallback)</span>", unsafe_allow_html=True)
credit_model = None
if credit_file:
    try:
        credit_model = joblib.load(credit_file)
        st.sidebar.success("Credit model loaded.")
    except Exception as e:
        st.sidebar.error("Could not load credit model: " + str(e))

if credit_model is not None:
    try:
        X_sup = df[['amount']].fillna(0)
        sup_probs = credit_model.predict_proba(X_sup)[:,1]
        df['supervised_prob'] = sup_probs
        st.success("Supervised probabilities from uploaded credit model computed using 'amount' input.")
    except Exception:
        st.warning("Uploaded credit model failed on 'amount' input; using amount-based fallback.")
        df['supervised_prob'] = (df['amount'] / (df['amount'].max() + 1e-9)).clip(0,1)
else:
    df['supervised_prob'] = (df['amount'] / (df['amount'].max() + 1e-9)).clip(0,1)
    st.info("No credit model uploaded — using amount-normalized supervised fallback.")

# ---------------- Combine final score & label -------------------
df['final_score'] = sup_weight * df['supervised_prob'] + (1 - sup_weight) * df['anomaly_prob']
big_amt = df['amount'].quantile(0.995)
df.loc[df['amount'] > big_amt, 'final_score'] = np.minimum(1.0, df['final_score'] + 0.12)
df['risk_label'] = pd.cut(df['final_score'], bins=[-0.01,0.3,0.7,1.0], labels=['LOW','MEDIUM','HIGH'])

# ---------------- KPIs -------------------
st.markdown("## <span class='subtitle'>Key metrics</span>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", len(df))
k2.metric("High risk", int((df['risk_label']=='HIGH').sum()))
k3.metric("Anomalies", int(df['anomaly_flag'].sum()))
k4.metric("Avg score", f"{df['final_score'].mean():.2f}")

st.markdown("---")

# ---------------- Visualizations -------------------
left, right = st.columns([2,1])

with left:
    st.subheader("Final Score Distribution")
    if 'isFraud' in df.columns:
        fraud_scores = df[df['isFraud']==1]['final_score']
        notfraud_scores = df[df['isFraud']==0]['final_score']
        chart_df = pd.DataFrame({
            'score': np.concatenate([fraud_scores, notfraud_scores]),
            'label': ['Fraud']*len(fraud_scores) + ['Not Fraud']*len(notfraud_scores)
        })
        chart = alt.Chart(chart_df).mark_bar(opacity=0.7).encode(
            alt.X('score:Q', bin=alt.Bin(maxbins=60), title='Final score'),
            y='count()',
            color='label:N'
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.altair_chart(alt.Chart(pd.DataFrame({'score':df['final_score']})).mark_bar().encode(alt.X('score:Q', bin=True), y='count()').properties(height=300), use_container_width=True)

    st.subheader("Top High-Risk Transactions")
    st.dataframe(df.sort_values('final_score', ascending=False).head(20)[['amount','supervised_prob','anomaly_prob','final_score','risk_label']])

with right:
    st.subheader("Risk breakdown")
    breakdown = df['risk_label'].value_counts().rename_axis('risk').reset_index(name='count')
    st.bar_chart(breakdown.set_index('risk'))

# ---------------- If labels available: performance -------------------
if 'isFraud' in df.columns:
    st.markdown("### <span class='subtitle'>Model performance (using isFraud)</span>", unsafe_allow_html=True)
    df['pred'] = (df['final_score'] >= 0.5).astype(int)

    acc = accuracy_score(df['isFraud'], df['pred'])
    st.write(f"**Accuracy:** {acc:.3f}")
    st.markdown("**Classification Report:**")
    st.text(classification_report(df['isFraud'], df['pred']))

    cm = confusion_matrix(df['isFraud'], df['pred'])
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(df['isFraud'], df['final_score'])
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#0b3d91")
    ax2.plot([0,1],[0,1],'k--')
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC curve")
    ax2.legend()
    st.pyplot(fig2)

    prec, rec, _ = precision_recall_curve(df['isFraud'], df['final_score'])
    fig3, ax3 = plt.subplots()
    ax3.plot(rec, prec, color="#d62828")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision"); ax3.set_title("Precision-Recall")
    st.pyplot(fig3)

# ---------------- Download results -------------------
st.markdown("### Download scored data")
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button("Download hybrid_scores.csv", data=to_csv_bytes(df), file_name="hybrid_scores.csv", mime="text/csv")

st.markdown("---")
st.markdown('<div class="footer">Tip: To reuse your anomaly model across runs, save the dict <code>{"iso":iso,"scaler":scaler,"features":features}</code> and upload it here or place it at <code>models/iso_model.pkl</code>.</div>', unsafe_allow_html=True)
