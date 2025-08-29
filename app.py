import streamlit as st
import os
import json
import openai
import faiss
import numpy as np
import requests

# 讀取 testdata.txt 並解析主旨與全文
def load_testdata():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'testdata.txt'))
    items = []
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 解析主旨
                if line.startswith('主旨：'):
                    parts = line.split('內容：', 1)
                    if len(parts) == 2:
                        subject = parts[0].replace('主旨：', '').replace('\n', '').strip()
                        content = parts[1].replace('\n', '').strip()
                        full = f"主旨：{subject}\n內容：{content}"
                        items.append({'subject': subject, 'content': content, 'full': full})
    return items

testdata_items = load_testdata()
testdata_subjects = [item['subject'] for item in testdata_items]

# 透過 st.secrets 讀取機密與設定
openai.api_key = st.secrets['OpenAI']['api_key']
openai.api_base = st.secrets['OpenAI']['base_url']

openrouter_api_key = st.secrets['OpenRouter']['api_key']
openrouter_api_base = st.secrets['OpenRouter']['base_url']
chat_model = st.secrets['OpenRouter']['model']

embedding_model = st.secrets['Embedding']['model']
vector_db_path = st.secrets['Embedding']['vector_db_path']

# 嵌入函式
def get_embedding(text: str) -> np.ndarray:
    resp = openai.embeddings.create(model=embedding_model, input=text)
    return np.array(resp.data[0].embedding, dtype='float32')

# 相似度搜尋
def search(query: str, field: str = 'qs', top_k: int = 5):
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), vector_db_path))
    idx_file = 'index_q.faiss' if field == 'qs' else 'index_a.faiss'
    meta_file = 'meta_q.json' if field == 'qs' else 'meta_a.json'
    idx = faiss.read_index(os.path.join(db_path, idx_file))
    with open(os.path.join(db_path, meta_file), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    emb = get_embedding(query)
    D, I = idx.search(emb[np.newaxis, :], top_k)
    results = []
    for d, i in zip(D[0], I[0]):
        item = meta[i].copy()
        item['距離'] = float(d)
        results.append(item)
    return results

# 組 prompt
def build_prompt(query, results):
    prompt = [
        "請參考下列最相似的來文及回復案例，撰寫符合公文格式的回文，並且參考最相似案例回答之邏輯、內容及引用的法條與說明進行回答，不在以下最相似案例中引用的法條不要使用，必須保持中立的立場進行撰寫，撰寫內容不可帶有主觀判定個案是否合理或是否合法，例如：不可以直接判斷公司某個行為是否超出合理範圍，但可以說明當發生什麼情況下會違反什麼法規，內容中有可參考到法條的必須明列且正確的列出參考是第幾條第幾項，法條文字內容需從最相似案例中進行參考，不可以自行生成，如果沒有可參考的法源依據則必須告知法無明文規定。",
        "--- 最相似案例 ---"
    ]
    for i, item in enumerate(results, 1):
        prompt.append(f"\n【來文{i}】\n主旨：{item['q'].split('說明：')[0].replace('主旨：','').strip()}\n說明：{item['q'].split('說明：')[1].strip()}\n【回文{i}】\n{item['a']}")
    prompt.append("\n--- 本次來文內容 ---")
    prompt.append(query)
    return '\n'.join(prompt)

# 呼叫 OpenRouter API
def chat_completion(prompt_text):
    url = f"{openrouter_api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            return "[錯誤] 無法取得回覆，請檢查 OpenRouter API key 與設定。"
        data = resp.json()
        if isinstance(data, dict) and data.get('choices'):
            return data['choices'][0].get('message', {}).get('content', '').strip()
        else:
            return "[錯誤] 無法解析 OpenRouter 回應，請檢查回傳格式。"
    except Exception:
        return "[錯誤] 呼叫 OpenRouter API 失敗。"

st.set_page_config(page_title='自動回覆系統', layout='wide')

st.title('自動回覆系統')


# 預設內容
default_text = testdata_items[0]['full'] if testdata_items else ''
if 'query_text' not in st.session_state:
    st.session_state['query_text'] = default_text

# 新增選單：選擇範例來文
def on_select_change():
    idx = st.session_state['testdata_select']
    if 0 <= idx < len(testdata_items):
        st.session_state['query_text'] = testdata_items[idx]['full']

st.markdown('**選擇範例來文自動帶入下方欄位：**')
testdata_idx = 0
if testdata_items:
    testdata_idx = st.selectbox('選擇範例主旨', options=range(len(testdata_subjects)), format_func=lambda i: testdata_subjects[i], key='testdata_select', on_change=on_select_change)

# 使用者輸入來文內容
query = st.text_area('請輸入來文內容', height=200, key='query_text')

# 選擇搜尋欄位
option_map = {'歷史來文 (qs)': 'qs', '歷史回文 (answer)': 'answer'}
display = st.selectbox('選擇搜尋欄位', options=list(option_map.keys()))
field = option_map[display]

# 選擇 top-k
top_k = st.number_input('選擇 Top-k', min_value=1, max_value=10, value=3, step=1)

# 產生回覆按鈕
if st.button('產生回覆'):
    if not query.strip():
        st.warning('請輸入來文內容後再按下產生回覆')
    else:
        with st.spinner('正在搜尋相似案例並呼叫語言模型...'):
            results = search(query, field=field, top_k=top_k)
            prompt = build_prompt(query, results)
            reply = chat_completion(prompt)
        # 顯示回覆結果
        st.subheader('回覆結果')
        st.text_area('語言模型 回覆', value=reply, height=300)
        # 顯示完整 Prompt
        st.subheader('組成的 Prompt')
        st.code(prompt, language='text')
