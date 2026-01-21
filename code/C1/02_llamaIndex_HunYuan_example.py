import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from tencent_hunyuan_embeddings import HunYuanEmbeddings

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 初始化混元 Embedding（替换原来的 HuggingFaceEmbeddings）
embeddings = HunYuanEmbeddings(
    secret_id=os.getenv("SecretId", ""),
    secret_key=os.getenv("SecretKey", ""),
    region="ap-guangzhou"
)

# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)

# 检查检索结果是否为空
if not retrieved_docs:
    print("未找到相关文档，无法回答问题。")
else:
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 配置混元 API（从环境变量读取）
    client = OpenAI(
        api_key=os.getenv("HUNYUAN_API_KEY", ""),
        base_url="https://api.hunyuan.cloud.tencent.com/v1",
    )

    completion = client.chat.completions.create(
        model="hunyuan-turbos-latest",
        messages=[
            {
                "role": "user",
                "content": prompt.format(question=question, context=docs_content)
            }
        ],
        extra_body={
            "enable_enhancement": True,  # <- 自定义参数
        },
    )
    print(completion.choices[0].message.content)