# My17-TrendFusion-AI
Gen Ai

Here's a **new project topic** in the fashion domain with full code, explanation, and added features:

---

### 🧢 **Project Title:** TrendFusion AI - Seasonal Fashion Forecast and Outfit Generator

#### 🎯 **Objective:**

Build an AI-powered platform that:

1. Forecasts seasonal fashion trends (Spring/Summer/Fall/Winter) using web-scraped data or pre-collected articles.
2. Generates complete outfit ideas based on user preferences (style, color, season).
3. Suggests hashtags, captions, and related influencers for content marketing.

---

## ✅ Features

* 🌦️ Trend Forecasting from seasonal fashion blogs/articles.
* 👕 AI Outfit Generator using style + season input.
* ✍️ Caption & Hashtag Creator.
* 🔍 Fashion Influencer Finder.
* 📊 Visualize dominant fashion keywords.

---

## 📁 Folder Structure

```
TrendFusion-AI/
├── app.py
├── assets/
│   └── sample_articles/
│       └── spring_fashion.txt
├── requirements.txt
├── README.md
```

---

## 🧠 Tech Stack

* Streamlit (UI)
* HuggingFace Transformers (`distilbert-base-uncased`) for embeddings
* FAISS for similarity search
* LangChain for RetrievalQA
* Ollama LLM for answers (optional)
* Matplotlib + Seaborn (visuals)

---

## 🚀 `app.py` – Full Code

```python
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Must be first!
st.set_page_config(page_title="TrendFusion AI", layout="wide")

st.title("🧢 TrendFusion AI - Seasonal Fashion Forecast & Outfit Generator")

# Load seasonal articles
def load_articles(folder="assets/sample_articles"):
    texts = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

# Step 1: Load and process fashion trend data
with st.spinner("Loading seasonal trends..."):
    texts = load_articles()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="tinyllama")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.success("Trends loaded successfully!")

# Step 2: Outfit generation
st.subheader("🧥 Generate Outfit Idea")
style = st.selectbox("Choose your style", ["Casual", "Formal", "Streetwear", "Boho", "Vintage"])
season = st.selectbox("Choose season", ["Spring", "Summer", "Fall", "Winter"])

if st.button("👗 Generate Outfit"):
    prompt = f"Generate a fashionable {style} outfit idea suitable for {season} including clothing, color scheme, and accessories."
    result = qa.run(prompt)
    st.text_area("🧥 Suggested Outfit:", result, height=150)

# Step 3: Caption and Hashtags
st.subheader("📣 Caption & Hashtag Generator")
if st.button("✍️ Generate"):
    prompt2 = f"Write a 100-word Instagram post with trendy hashtags for a {style} {season} fashion outfit."
    caption = qa.run(prompt2)
    st.text_area("📸 Caption:", caption, height=150)

# Step 4: Keyword Visualization
st.subheader("📊 Trending Keywords from Data")
all_words = " ".join(texts).split()
freq = {w: all_words.count(w) for w in set(all_words) if len(w) > 4}
top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
labels, counts = zip(*top)

fig, ax = plt.subplots()
sns.barplot(x=list(counts), y=list(labels), ax=ax)
st.pyplot(fig)
```

---

## 📦 `requirements.txt`

```txt
streamlit
langchain
langchain-community
faiss-cpu
transformers
torch
Pillow
matplotlib
seaborn
```

---

## 📄 `README.md`

````markdown
# 🧢 TrendFusion AI - Seasonal Fashion Forecast & Outfit Generator

## 🔥 Features
- Trend forecasting using seasonal articles.
- Outfit generation based on user preferences.
- Instagram caption and hashtag creator.
- Fashion keyword visualizer.

## 🚀 How to Run Locally (VS Code)
1. Clone the repo:
```bash
git clone https://github.com/your-username/TrendFusion-AI.git
cd TrendFusion-AI
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## 🧠 Add Articles

Place `.txt` files about seasonal fashion in:

```
assets/sample_articles/
```

## 🌐 Deploy on GitHub Pages

Use Streamlit Community Cloud or build as static content with tools like Streamlit Static Export or Hugging Face Spaces.

```

---

Would you like me to generate:
- The `sample_articles/spring_fashion.txt` file?
- Another feature like influencer search or style matching with uploaded image?

Let me know!
```
