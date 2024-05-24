# MoPE: Mixture of Prefix Experts for Zero-Shot Dialogue State Tracking

## Abstract

Zero-shot dialogue state tracking (DST) transfers knowledge to unseen domains, reducing the cost of annotating new datasets. Previous zero-shot DST models mainly suffer from domain transferring and partial prediction problems. To address these challenges, we propose Mixture of Prefix Experts (MoPE) to establish connections between similar slots in different domains, which strengthens the model transfer performance in unseen domains. Empirical results demonstrate that MoPE-DST achieves the joint goal accuracy of 57.13% on MultiWOZ2.1 and 55.40% on SGD.

## Model Architecture

![](./img/structure.png)
Illustration of our proposed method, including (a) Slot clustering, (b) Deep Prefix Prompt Tuning, and (c) Multiple
Prefix Prompt Generation. Slot clustering is used to categorize all slots into distinct clusters and establishes
connections between slots in different domains. Deep Prefix Prompt Tuning is our method to strengthen the LLM's
conditional generation. Multiple Prefix Prompt Generation shows the complete pipeline of solving DST task.

## Depenedency

### Download packages

`python >= 3.10`

```bash
pip install -r requirements.txt
```

### Download pretrained model

[chatglm-6b download link](https://huggingface.co/THUDM/chatglm-6b)

### Download & process source data

```shell
bash download_process_data.sh
```
## Train & Evaluate

```bash
bash run.sh
```

**note:** 

`--chatglm_path` should be changed to your personal pretrained model save path.

`--result_csv_dir` is the directory of the saved results. We provide two display forms for the results, you can refer to the csv file to check the main results. Further more, we provide a sqlite3 db to record all results during the whole train and evaluation, such as the performance on the validation data and so on. 

`--checkpoint` all tuned parameters are stored in `save/${dataset}/${exclude_domain}/${train_id}/${feature}/best`, so you can get the whole information of the checkpoint in the directory mentioned above.


# Other notes

we find that there exists some variances during the train. Therefore, there may be some fluctuations in the experimental results.

# Citation

If you think that our work is helpful to you, don't forget to cite us.

```
@inproceedings{tang-etal-2024-mope-mixture,
    title = "{M}o{PE}: Mixture of Prefix Experts for Zero-Shot Dialogue State Tracking",
    author = "Tang, Tianwen  and
      Zhu, Tong  and
      Liu, Haodong  and
      Bai, Yin  and
      Cheng, Jia  and
      Chen, Wenliang",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1012",
    pages = "11582--11592",
    abstract = "Zero-shot dialogue state tracking (DST) transfers knowledge to unseen domains, reducing the cost of annotating new datasets. Previous zero-shot DST models mainly suffer from domain transferring and partial prediction problems. To address these challenges, we propose Mixture of Prefix Experts (MoPE) to establish connections between similar slots in different domains, which strengthens the model transfer performance in unseen domains. Empirical results demonstrate that MoPE-DST achieves the joint goal accuracy of 57.13{\%} on MultiWOZ2.1 and 55.4.",
}
```
