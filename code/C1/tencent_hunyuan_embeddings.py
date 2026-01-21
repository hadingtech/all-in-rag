import os
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
import numpy as np
from typing import List


class HunYuanEmbeddings:
    """腾讯混元 Embedding 模型封装，对齐 HuggingFaceEmbeddings 的接口"""
    
    def __init__(self, secret_id: str, secret_key: str, region: str = "ap-guangzhou"):
        """
        初始化腾讯混元 Embedding
        :param secret_id: 腾讯云 SecretId
        :param secret_key: 腾讯云 SecretKey
        :param region: 地域，默认 ap-guangzhou
        """
        # 初始化腾讯云认证
        self.cred = credential.Credential(secret_id, secret_key)
        self.client = hunyuan_client.HunyuanClient(self.cred, region)
        self.normalize_embeddings = True  # 对应你原来的 normalize_embeddings=True
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入（对应 HuggingFaceEmbeddings 的 embed_documents 方法）
        :param texts: 文本列表
        :return: 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            try:
                req = models.GetEmbeddingRequest()
                req.Input = text
                # 调用混元 Embedding 接口
                resp = self.client.GetEmbedding(req)
                vec = resp.Data[0].Embedding
                # 归一化（对应你原来的 normalize_embeddings=True）
                if self.normalize_embeddings:
                    vec = self._normalize_vector(vec)
                embeddings.append(vec)
            except TencentCloudSDKException as err:
                print(f"生成嵌入失败：{err}")
                embeddings.append([])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成单条查询文本的嵌入（对应 HuggingFaceEmbeddings 的 embed_query 方法）
        :param text: 查询文本
        :return: 嵌入向量
        """
        return self.embed_documents([text])[0]
    
    def _normalize_vector(self, vec: List[float]) -> List[float]:
        """向量归一化（L2 归一化）"""
        vec_np = np.array(vec)
        norm = np.linalg.norm(vec_np)
        if norm == 0:
            return vec_np.tolist()
        return (vec_np / norm).tolist()