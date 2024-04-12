import torch
from utils.args import args
from utils.logger import run_logger
from accelerate import Accelerator
from torch.optim import AdamW
from utils.data import data_loader
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate_dev_cluster
from utils.track import get_track
import os


def run(args):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    accelerate = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    args.num_gpu = accelerate.num_processes

    if args.backbone == "chatglm":
        from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
        from models.chatglm_generate import DST
        tokenizer = ChatGLMTokenizer.from_pretrained(args.chatglm_path)
    else:
        raise ValueError("please give right backbone of model")

    tokenizer.truncation_side = "left"


    train_loader = data_loader(args, tokenizer, "train")
    dev_loader = data_loader(args, tokenizer, "dev")

    model = DST(args, tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model = accelerate.prepare_model(model)
    optimizer = accelerate.prepare_optimizer(optimizer)
    train_loader = {k: accelerate.prepare_data_loader(v) for k, v in train_loader.items()}

    model = model.module if hasattr(model, "module") else model

    if args.plot_loss:
        run_logger.info(
            f"plot loss path: ./runs/{args.dataset}-{args.exclude_domain}-{args.train_id}-{args.cluster_feature}"
        )

        if os.path.exists(f"./runs/{args.dataset}-{args.exclude_domain}-{args.train_id}-{args.cluster_feature}"):
            os.system(f"rm -rf ./runs/{args.dataset}-{args.exclude_domain}-{args.train_id}-{args.cluster_feature}")

        writer = SummaryWriter(
            f"./runs/{args.dataset}-{args.exclude_domain}-{args.train_id}-{args.cluster_feature}"
        )

    datas = {
        k: iter(v) for k, v in train_loader.items()
    }
    dev_datas = {
        k: v for k, v in dev_loader.items()
    }

    cluster_ids = dev_loader.keys()

    if args.zero_shot and args.dataset == "sgd":
        assert set(dev_loader.keys()) == set(train_loader.keys())

    best_slot_acc = {
        k: -1 for k in cluster_ids
    }

    tmp_loss = {
        k: 1e18 for k in cluster_ids
    }

    early_stop = {
        k: 0 for k in cluster_ids
    }

    progress = get_track()
    progress.start()

    total_track = progress.add_task("total step: ", total=args.total_step)

    train_track = progress.add_task("train")
    dev_track = progress.add_task("dev")

    for cluster_id in cluster_ids:
        for step in range(0, args.total_step, args.save_step):
            if early_stop[cluster_id] >= args.stop_time:
                continue
            progress.reset(train_track, description=f"train cluster-{cluster_id}: ", total=args.save_step)

            all_loss = []

            for cluster_step in range(args.save_step):
                try:
                    data = next(datas[cluster_id])
                except StopIteration:
                    datas[cluster_id] = iter(train_loader[cluster_id])
                    data = next(datas[cluster_id])
                with accelerate.accumulate(model):
                    loss = model(data)
                    accelerate.backward(loss)
                    optimizer.step()
                    model.zero_grad()
                    if args.plot_loss and (cluster_step + 1) % 5 == 0:
                        writer.add_scalar(
                            f"loss-{cluster_id}",
                            loss.item(),
                            step + cluster_step
                        )
                    all_loss.append(loss.item())


                progress.advance(train_track, 1)
            if accelerate.is_local_main_process:

                if args.stop_metrics == "loss":
                    mean_loss = sum(all_loss) / len(all_loss)
                    if mean_loss < tmp_loss[cluster_id]:
                        tmp_loss[cluster_id] = mean_loss
                        early_stop[cluster_id] = 0
                        unwrap_model = accelerate.unwrap_model(model)
                        save_dir = f"{args.dataset}/{args.exclude_domain}/{args.train_id}/{args.cluster_feature}/best"
                        unwrap_model.save_model(
                            save_dir
                        )
                    else:
                        early_stop[cluster_id] += 1
                elif args.stop_metrics == "slot_acc":
                    slot_acc = evaluate_dev_cluster(
                        args,
                        dev_datas[cluster_id],
                        model,
                        tokenizer,
                        step + args.save_step,
                        cluster_id,
                        progress,
                        dev_track
                    )
                    if best_slot_acc[cluster_id] < slot_acc and args.save_model:
                        best_slot_acc[cluster_id] = slot_acc
                        early_stop[cluster_id] = 0
                        unwrap_model = accelerate.unwrap_model(model)
                        save_dir = f"{args.dataset}/{args.exclude_domain}/{args.train_id}/{args.cluster_feature}/best"
                        unwrap_model.save_model(
                            save_dir
                        )
                    else:
                        early_stop[cluster_id] += 1
                else:
                    unwrap_model = accelerate.unwrap_model(model)
                    save_dir = f"{args.dataset}/{args.exclude_domain}/{args.train_id}/{args.cluster_feature}/best"
                    unwrap_model.save_model(
                        save_dir
                    )
            accelerate.wait_for_everyone()

        progress.advance(total_track, args.save_step)
        stop_cnt = 0
        for k, v in early_stop.items():
            stop_cnt += v
        if stop_cnt == len(cluster_ids) * args.stop_time:
            print("early stop.............")
            print(best_slot_acc)
            break


if __name__ == "__main__":
    run_logger.info(f"dataset: {args.dataset}")
    run_logger.info(
        f"{f'zero shot: exclude domain {args.exclude_domain}' if args.zero_shot else f'few shot: {args.data_ratio}'}")
    run_logger.info(f"backbone: {args.backbone}")
    run_logger.info(f"num sample: {args.num_sample}")
    run_logger.info(f"train batch size: {args.train_batch_size}")
    run_logger.info(f"save step: {args.save_step}")
    run_logger.info(f"n_clusters: {args.n_clusters}")
    run_logger.info(f"feature: {args.cluster_feature}")
    run_logger.info(f"num gpus: {torch.cuda.device_count()}")
    run(args)
