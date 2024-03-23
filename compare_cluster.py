import os.path

import pandas as pd
import json
from sklearn.cluster import KMeans
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import numpy as np
from sklearn.metrics import silhouette_score


def compare_cluster_with_transformer(dataset, slotnames, tokenizer, ptm):
    encoded = tokenizer(slotnames, padding=True)
    for k in encoded.keys():
        encoded[k] = torch.tensor(np.array(encoded[k]))
    encoded = encoded.to("cuda")

    output = ptm(**encoded)

    x = output[0].flatten(1, 2)
    x = x.cpu().detach().numpy()

    SSE = {"n_clusters": [], "SSE": []}
    silhouette = {"n_clusters": [], "silhouette": []}

    for n_clusters in range(1, 10):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(x)
        SSE["n_clusters"].append(n_clusters)
        SSE["SSE"].append(kmeans.inertia_)
        if n_clusters > 1:
            silhouette["n_clusters"].append(n_clusters)
            silhouette["silhouette"].append(silhouette_score(x, kmeans.labels_))

    save_dir = "./result/cluster"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    SSE = pd.DataFrame(SSE)
    print(SSE)
    SSE.to_csv(os.path.join(save_dir, f"SSE-transformer-{dataset}.csv"))

    silhouette = pd.DataFrame(silhouette)
    print(silhouette)
    silhouette.to_csv(os.path.join(save_dir, f"silhouette-transformer-{dataset}.csv"))


def compare_cluster_with_embedding(dataset, slotnames, tokenizer, ptm):
    encoded = tokenizer(slotnames, padding=True)
    for k in encoded.keys():
        encoded[k] = torch.tensor(np.array(encoded[k]))
    encoded = encoded.to("cuda")

    embedding = ptm.transformer.get_input_embeddings()

    output = embedding(encoded["input_ids"])

    x = output.flatten(1, 2).cpu().detach().numpy()

    SSE = {"n_clusters": [], "SSE": []}
    silhouette = {"n_clusters": [], "silhouette": []}

    for n_clusters in range(1, 10):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(x)
        SSE["n_clusters"].append(n_clusters)
        SSE["SSE"].append(kmeans.inertia_)
        if n_clusters > 1:
            silhouette["n_clusters"].append(n_clusters)
            silhouette["silhouette"].append(silhouette_score(x, kmeans.labels_))

    save_dir = "./result/cluster"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    SSE = pd.DataFrame(SSE)
    print(SSE)
    SSE.to_csv(os.path.join(save_dir, f"SSE-embedding-{dataset}.csv"))

    silhouette = pd.DataFrame(silhouette)
    print(silhouette)
    silhouette.to_csv(os.path.join(save_dir, f"silhouette-embedding-{dataset}.csv"))


def load_slotnames(dataset):
    path = f"./data/{dataset}/schema.json"
    assert os.path.exists(path), print(f"{path} is not exist")

    schema = json.load(open(path, "r"))

    domains = []
    slotnames = []
    for k, v in schema.items():
        slotnames.extend(v)
        domains.append(k)
    return domains, slotnames


if __name__ == "__main__":
    model_path = "/public/home/wlchen/plm/chatglm-6b"
    tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    ptm = ChatGLMForConditionalGeneration.from_pretrained(model_path).half().cuda()
    datasets = ["crosswoz", "sgd", "multiwoz2.2"]

    for dataset in datasets:
        _, slotnames = load_slotnames(dataset)
        compare_cluster_with_embedding(dataset, slotnames, tokenizer, ptm)
        compare_cluster_with_transformer(dataset, slotnames, tokenizer, ptm)

    few_shot_datasets = ["multiwoz2.1", "multiwoz2.2"]

    for dataset in few_shot_datasets:
        domains, slotnames = load_slotnames(dataset)
        for domain in domains:
            tmp_slotnames = [slotname for slotname in slotnames if domain not in slotname]
            compare_cluster_with_embedding(f"{dataset}-{domain}", tmp_slotnames, tokenizer, ptm)
            compare_cluster_with_transformer(f"{dataset}-{domain}", tmp_slotnames, tokenizer, ptm)
