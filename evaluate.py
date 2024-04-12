import torch
from utils.args import args
from utils.data import data_loader
from utils.logger import run_logger
from utils.track import get_track
from utils.metric import ValueMetric


@torch.no_grad()
def evaluate(args, loader, model, tokenizer, progress, task_id, **kwargs):
    metric = ValueMetric(args, tokenizer)
    model.eval()
    not_none_data = []
    for idx, data in enumerate(loader):
        predict, d = model.infer(data)
        for i in range(len(predict)):
            # if predict[i] != "none" or d["labels"][i] != "none":
            not_none_data.append({
                "dialogue_id": d["dialogue_id"][i],
                "turn": d["turn"][i],
                "domain": d["domain"][i],
                "slotname": d["slotname"][i],
                "input_context": d["input_context"][i],
                "predict_state": predict[i],
                "labels": d["labels"][i]
            })
        progress.advance(task_id, 1)

    metric.add(not_none_data)
    metric.compute(model.uuid)


@torch.no_grad()
def evaluate_dev_cluster(args, loader, model, tokenizer, step, cluster_id, progress, task_id, **kwargs):
    metric = ValueMetric(args, tokenizer)
    model.eval()
    not_none_data = []
    progress.reset(task_id, total=len(loader))
    for idx, data in enumerate(loader):
        predict, d = model.infer(data)
        for i in range(len(predict)):
            # if predict[i] != "none" or d["labels"][i] != "none":
            not_none_data.append({
                "dialogue_id": d["dialogue_id"][i],
                "turn": d["turn"][i],
                "domain": d["domain"][i],
                "slotname": d["slotname"][i],
                "input_context": d["input_context"][i],
                "predict_state": predict[i],
                "labels": d["labels"][i]
            })
        progress.advance(task_id, 1)

    metric.add(not_none_data)
    metric.compute(model.uuid, step, cluster_id)
    return metric.slot_acc


def run(args):
    if args.backbone == "chatglm":
        from models.chatglm_generate import DST
        from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
        tokenizer = ChatGLMTokenizer.from_pretrained(args.chatglm_path)
    else:
        raise ValueError("backbone is not correct")

    test_loader = data_loader(args, tokenizer, "test")

    model = DST(args, tokenizer)

    model = model.cuda()

    progress = get_track()
    progress.start()
    evaluate_cluster = progress.add_task("evaluate cluster", total=len(test_loader))

    evaluate(args, test_loader, model, tokenizer, progress, evaluate_cluster)


if __name__ == "__main__":
    run_logger.info(f"dataset: {args.dataset}")
    run_logger.info(
        f"{f'zero shot: exclude domain {args.exclude_domain}' if args.zero_shot else f'few shot: {args.data_ratio}'}")
    run_logger.info(f"num sample: {args.num_sample}")
    run_logger.info(f"checkpoint: {args.checkpoint}")
    run_logger.info(f"train_id : {args.train_id}")
    run_logger.info(f"n_clusters: {args.n_clusters}")
    run_logger.info(f"feature: {args.cluster_feature}")
    run(args)
