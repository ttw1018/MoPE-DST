import os.path
import torch
import numpy as np
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from sklearn.cluster import KMeans
import joblib
from utils.logger import run_logger
import pandas as pd


def cluster(args, tokenizer, slotnames, all_slotnames, run_type="train"):

    n_clusters = args.n_clusters

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
        ptm = ptm.transformer
        output = ptm(**encoded)
        x = output[0].permute(1, 0, 2).flatten(1, 2)
    elif args.cluster_feature == "embedding":
        embedding = ptm.transformer.get_input_embeddings()
        output = embedding(encoded["input_ids"])
        x = output.flatten(1, 2)
    else:
        raise ValueError(f"cluster_feature {args.cluster_feature} is not corrrect")

    cosine_similarity_matrix = []
    cos = torch.nn.CosineSimilarity(dim=0)
    for i in range(x.shape[0]):
        cosine_sim_matrix = []
        for j in range(x.shape[0]):
            cosine_sim_matrix.append(cos(x[i].float(), x[j].float()).item())
        cosine_similarity_matrix.append(cosine_sim_matrix)

    x = x.cpu().detach().numpy()

    save_dir = f"{args.dataset}/" \
               f"{args.exclude_domain if args.zero_shot else args.data_ratio}/{args.train_id}/{args.cluster_feature}"

    save_dir = os.path.join(args.save_dir, save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"kmeans-{n_clusters}.pkl")

    if run_type == "train":
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=args.random_seed)
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

    pout = open(os.path.join(save_dir, f"{n_clusters}-cluster-{run_type}-heatmap.txt"), "w")

    source_heatmap = []

    pos = [[i, int(y[i])] for i in range(len(slotnames))]
    pos.sort(key=lambda x: x[1])
    cluster_similarity = []
    sum1, sum2 = 0, 0
    cnt1, cnt2 = 0, 0
    for i in range(len(slotnames)):
        for j in range(len(slotnames)):
            if pos[i][1] == pos[j][1]:
                cluster_similarity.append([i, j, cosine_similarity_matrix[i][j]])
                sum2 += cosine_similarity_matrix[i][j]
                cnt2 += 1
            else:
                cluster_similarity.append([i, j, 0])
            source_heatmap.append([i, j, cosine_similarity_matrix[i][j]])
            if i != j:
                sum1 += cosine_similarity_matrix[i][j]
                cnt1 += 1

    cluster_slotname = []
    for i in pos:
        cluster_slotname.append(slotnames[i[0]])

    print(f"{sum1 / cnt1 : .4f} {sum2 / cnt2 : .4f}", file=pout)
    print(cluster_slotname, file=pout)
    print(source_heatmap, file=pout)
    print(cluster_similarity, file=pout)

    return result
