import os
import io
import streamlit as st
from openai import OpenAI
import chromadb
from pypdf import PdfReader
import docx

# ===================== 【配置区】 =====================
API_KEY = st.secrets["DASHSCOPE_API_KEY"]
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-turbo"
EMBED_MODEL = "text-embedding-v1" # 通义千问嵌入模型

PERSIST_DIR = "./chroma_db" # 向量库持久化目录
DOC_FOLDER = "./knowledge" # 知识库 PDF 目录
COLLECTION_NAME = "lvjian_kb"

CHUNK_SIZE = 800 # 文本切块大小（字符）
CHUNK_OVERLAP = 100 # 块间重叠
TOP_K = 3 # 检索返回条数
EMBED_BATCH = 20 # 嵌入接口单次最大条数

# 关键修改：强制侧边栏展开，并设置宽屏模式
st.set_page_config(
    page_title="律简——大学生校园权益智能咨询助手",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 安全的美化代码（不会隐藏侧边栏，只改变颜色和按钮样式）
st.markdown("""
<style>
    /* 1. 隐藏默认的顶部菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 2. 美化主标题 */
    h1 {
        color: #1E3A8A; /* 深法律蓝 */
        font-family: 'Microsoft YaHei', sans-serif;
        font-weight: 700;
    }
    
    /* 3. 美化侧边栏（保留显示，仅改颜色） */
    [data-testid="stSidebar"] {
        background-color: #1E3A8A !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* 4. 美化按钮 */
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.title("律简｜大学生校园权益智能咨询助手")
st.info("⚠️ 本工具仅作为权益科普参考，不构成正式法律/行政意见")

# 全局 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY, 
    base_url=API_BASE_URL,
    timeout=30.0 #超过30秒自动断开
)

# ===================== 基础工具函数 =====================
def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)

def chat_with_history(prompt: str, history: list = None, system: str = None, temperature: float = 0.2) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        for user_msg, bot_msg in history[-3:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

# ===================== 向量库构建 / 加载 =====================
@st.cache_resource(show_spinner=False)
def get_collection():
    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        return collection

    if not os.path.isdir(DOC_FOLDER):
        return collection

    docs, ids, metas = [], [], []
    for fname in sorted(os.listdir(DOC_FOLDER)):
        if not fname.lower().endswith(".pdf"):
            continue
        try:
            text = read_pdf_text(os.path.join(DOC_FOLDER, fname))
        except Exception as e:
            st.warning(f"读取 {fname} 失败：{e}")
            continue
        for i, chunk in enumerate(split_text(text)):
            docs.append(chunk)
            ids.append(f"{fname}::{i}")
            metas.append({"source": fname})

    if not docs:
        return collection

    embeddings = embed_texts(docs)
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return collection

def retrieve(collection, query: str, k: int = TOP_K):
    if collection.count() == 0:
        return []
    q_vec = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for d, m in zip(docs, metas):
        out.append((d, (m or {}).get("source", "未知来源")))
    return out

# ===================== 初始化 =====================
if not API_KEY or API_KEY.startswith("sk-在这里"):
    st.error("请先配置 DASHSCOPE_API_KEY（环境变量或代码里直接填）。")
    st.stop()

with st.spinner("正在加载知识库..."):
    collection = get_collection()

if collection.count() == 0:
    st.warning("知识库为空：请把 PDF 放进 ./knowledge 目录后重启应用。")

SYSTEM_PROMPT = (
    "你是【律简】大学生校园权益助手。"
    "只根据提供的参考文档内容回答用户问题，不要编造。"
    "用通俗易懂的中文解释，回答末尾注明引用来源。"
    "如果参考文档里没有相关内容，直接说明知识库暂未收录。"
)

# ===================== 左侧侧边栏导航 =====================
st.sidebar.title(" 律简导航")
st.sidebar.info("大学生校园权益智能咨询助手")

menu_options = [" 权益问题咨询", " 上传 PDF 文件分析", " 申诉/协商文书生成"]
choice = st.sidebar.radio("请选择功能模块：", menu_options)

# ===================== 功能 1：知识库问答（带记忆） =====================
if choice == " 权益问题咨询":
    st.subheader(" 权益问题咨询")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("查看参考文档片段"):
                    for i, (txt, src) in enumerate(msg["sources"]):
                        st.write(f"**【片段{i+1}｜{src}】** {txt[:300]}...")

    # 底部聊天输入框
    user_query = st.chat_input("描述你的校园/实习权益问题...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        history_for_llm = [(m["content"], st.session_state.messages[i+1]["content"]) 
                           for i, m in enumerate(st.session_state.messages) 
                           if m["role"] == "user" and i+1 < len(st.session_state.messages)]

        with st.spinner("AI 检索知识库并生成回答..."):
            hits = retrieve(collection, user_query, TOP_K)
            
            if not hits:
                answer = "知识库暂未收录相关内容，无法回答。"
                sources = []
            else:
                context = "\n\n".join(f"【片段{i+1}｜来源：{src}】\n{txt}" for i, (txt, src) in enumerate(hits))
                prompt = f"""请依据下面的参考文档回答用户问题。

【参考文档】
{context}

【用户问题】
{user_query}

要求：
1. 用通俗语言解释，末尾注明引用来源；
2. 文档没有相关内容就直接说"知识库暂未收录"，不要编造。
3. 结合之前的对话历史理解用户的意图。"""
                answer = chat_with_history(prompt, history=history_for_llm, system=SYSTEM_PROMPT)
                sources = hits

            # 显示 AI 回答
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("查看参考文档片段"):
                        for i, (txt, src) in enumerate(sources):
                            st.write(f"**【片段{i+1}｜{src}】** {txt[:300]}...")
            
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ===================== 功能 2：上传 PDF 文件分析 =====================
elif choice == " 上传 PDF 文件分析":
    st.subheader(" 上传 PDF 文件分析")
    st.write("上传实习合同、学校通知 PDF，AI 帮你找问题。")
    
    upload_file = st.file_uploader("点击选择 PDF 文件", type="pdf")

    if upload_file is not None:
        temp_path = "temp_upload.pdf"
        with open(temp_path, "wb") as f:
            f.write(upload_file.getvalue())

        question_for_pdf = st.text_input(
            "针对该 PDF 你想问什么？（留空则默认分析风险条款）",
            value="帮我找出这份文件的风险条款和不公平内容",
        )

        if st.button("开始分析该文件"):
            with st.spinner("正在解析上传文件..."):
                try:
                    text = read_pdf_text(temp_path)
                except Exception as e:
                    st.error(f"PDF 解析失败：{e}")
                    text = ""

                if not text.strip():
                    st.error("未能从 PDF 中提取到文本（可能是扫描件/图片型 PDF）。")
                else:
                    context = text[:6000]
                    prompt = f"""你是一名校园权益顾问。请依据下面的文件内容回答用户问题，
用通俗语言说明，并在末尾标注依据的原文片段。

【文件内容】
{context}

【用户问题】
{question_for_pdf}"""
                    answer = chat_with_history(prompt, system=SYSTEM_PROMPT)
                    st.write("### 分析结果：")
                    st.write(answer)

# ===================== 功能 3：文书草稿生成 =====================
elif choice == "✍️ 申诉/协商文书生成":
    st.subheader("✍️ 申诉/协商文书生成")
    
    brief_info = st.text_area(
        "填写你的情况，自动生成文书草稿：",
        placeholder="例如：奖学金评审存在异议，希望提交申诉",
        height=150,
    )

    if st.button("生成文书草稿") and brief_info.strip():
        with st.spinner("正在生成文书..."):
            prompt = f"""基于校园学籍、奖学金相关制度，根据下面用户情况，
生成一份简洁正式的申诉/协商文书草稿。
只输出文书正文，不要多余解释。

用户情况：{brief_info}"""
            draft = chat_with_history(prompt, temperature=0.3)
            st.write("### 文书草稿：")
            st.write(draft)
            doc = docx.Document()
            doc.add_heading('维权申诉书草稿', 0)
            for para in draft.split('\n'):
                if para.strip():
                    doc.add_paragraph(para)
            bio = io.BytesIO()
            doc.save(bio)
            st.download_button(
                label=" 下载正式 Word 文档 (.docx)",
                data=bio.getvalue(),
                file_name="维权申诉书草稿.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
