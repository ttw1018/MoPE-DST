from utils.args import args
from mydata.dst import data_loader
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer


def analysis(args, tokenizer):
    data = data_loader(args, tokenizer, "train")
    l1, l2 = [], []

    for d in data:
        for s in d["input_context"]:
            l1.append(len(tokenizer.encode(s)))
        # for s in d["labels"]:
        #     l2.append(len(tokenizer.encode(s)))

    cnt1 = 0
    for i in l1:
        if i > 384:
            cnt1 = cnt1 + 1

    # cnt2 = 0
    # for i in l2:
    #     if i > 32:
    #         cnt2 = cnt2 + 1
    #
    # cnt3 = 0
    # for i in l2:
    #     if i > 16:
    #         cnt3 = cnt3 + 1

    print(
        f"{args.datasets} history turn: {args.history_turn} max length: {max(l1)} mean length: {sum(l1) / len(l1)} longer 128: {cnt1 / len(l1)}")

    # print(
    #     f"{args.dataset} history turn: {args.history_turn} max length: {max(l2)} mean length: {sum(l2) / len(l2)} longer 32: {cnt2 / len(l2)} longer 16: {cnt3 / len(l2)}")

if __name__ == "__main__":
    args.chatglm_path = "/public/home/wlchen/plm/chatglm-6b"
    args.history_turn = 4
    args.use_single_state = True
    tokenizer = ChatGLMTokenizer.from_pretrained(args.chatglm_path)
    datasets = ["crosswoz", "risawoz", "sgd", "multiwoz2.0", "multiwoz2.1", "multiwoz2.2"]
    for dataset in datasets:
        args.datasets = dataset
        analysis(args, tokenizer)
