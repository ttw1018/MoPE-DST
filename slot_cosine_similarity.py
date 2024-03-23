import os.path
import torch
import numpy as np
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from sklearn.cluster import KMeans
import joblib
from utils.logger import run_logger
import pandas as pd


def cluster(args, slotnames, all_slotnames, run_type="train"):
    n_clusters = args.n_clusters
    tokenizer = ChatGLMTokenizer.from_pretrained(args.chatglm_path)

    diff = set(slotnames).difference(set(all_slotnames))
    assert len(diff) == 0, print("this is some slotnames is not in schema", diff)

    all_slotnames_encoded = tokenizer(all_slotnames)

    max_length = max([len(i) for i in all_slotnames_encoded.input_ids])

    encoded = tokenizer(slotnames, padding="max_length", max_length=max_length)
    for k in encoded.keys():
        encoded[k] = torch.tensor(np.array(encoded[k]))
    encoded = encoded.to("cuda")

    ptm = ChatGLMForConditionalGeneration.from_pretrained(args.chatglm_path).half().cuda()

    if args.cluster_feature == "transformer":
        output = ptm(**encoded)
        x = output[0].flatten(1, 2)
    elif args.cluster_feature == "embedding":
        embedding = ptm.transformer.get_input_embeddings()
        output = embedding(encoded["input_ids"])
        x = output.flatten(1, 2)
    else:
        raise ValueError(f"cluster_feature {args.cluster_feature} is not corrrect")

    x = x.cpu().detach().numpy()

    if args.zero_shot:
        save_dir = f"{args.dataset}/{args.exclude_domain}/{args.history_turn}/{args.train_id}/{args.cluster_feature}"
    else:
        save_dir = f"{args.dataset}/{args.data_ratio}/{args.history_turn}/{args.train_id}/{args.cluster_feature}"

    save_dir = os.path.join(args.save_dir, save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"kmeans-{n_clusters}.pkl")

    if run_type == "train":
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(x)
        y = kmeans.labels_
        joblib.dump(kmeans, save_path)
        run_logger.info(f"cluster {run_type} create model path: {save_path}")
    else:
        run_logger.info(f"cluster {run_type} load model path: {save_path}")
        kmeans = joblib.load(save_path)
        y = kmeans.predict(x)

    result = {slotnames[i]: int(y[i]) for i in range(len(slotnames))}

    cluster_slotname = pd.DataFrame({"slotname": slotnames, "cluster": y}).set_index("slotname").sort_values("cluster")

    cluster_slotname.to_csv(os.path.join(save_dir, f"{n_clusters}-cluster-{run_type}.csv"))

    return result
