import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
from utils.cluster import cluster


class DSTState(Dataset):
    def __init__(self, args, data, example, tokenizer, run_type="train"):
        super(DSTState, self).__init__()
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.run_type = run_type
        self.example = example

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return dict(self.data.iloc[item])

    def get_input_context(self, utterance, slotname, value):
        s = f"#dialogue: {utterance} #question: {slotname} #answer: {value}"

        if self.args.use_lower:
            s = s.lower()
        return s

    def get_example(self, slotname):
        examples = self.example[slotname].sample(self.args.num_sample, replace=True)
        outputs = []
        for k, v in examples.iterrows():
            s = self.get_input_context(v["utterance"], v["slotname"], v["value"])
            outputs.append(s)
        return outputs

    def collate_fn(self, datas):
        d = {
            "dialogue_id": [],
            "turn": [],
            "input_context": [],
            "labels": [],
            "slotname": [],
            "domain": [],
            "cluster_id": [],
            "inputs": {
            },
        }
        cluster = []

        for data in datas:
            d["slotname"].append(data["slotname"])
            d["turn"].append(data["turn"])
            d["labels"].append(data["value"])
            d["dialogue_id"].append(data["dialogue_id"])
            d["domain"].append(data["domain"])
            d["cluster_id"].append(data["cluster_id"])
            if self.args.random_prompt:
                cluster.append(random.randint(0, self.args.n_clusters - 1))
            else:
                cluster.append(data["cluster_id"])

        max_input_length = 0
        train_input = []
        masked_len = []

        for data in datas:
            utterance = data["utterance"]
            slotname = data["slotname"]
            value = data["value"]

            input_context = self.get_input_context(utterance, slotname, "")

            if self.args.num_sample > 0:
                examples = self.get_example(data["slotname"])
                examples.append(input_context)
                input_context = "\n".join(examples)

            d["input_context"].append(input_context)

            if self.run_type == "train":
                encoded_input = self.tokenizer.encode(input_context, add_special_tokens=False)
                encoded_label = self.tokenizer.encode(value, add_special_tokens=False)
                input_ids = self.tokenizer.build_inputs_with_special_tokens(encoded_input, encoded_label)
                if len(input_ids) > self.args.max_seq_length:
                    input_ids = input_ids[-self.args.max_seq_length:]
                max_input_length = max(max_input_length, len(input_ids))
                masked_len.append(len(input_ids) - 2 - len(encoded_label))
                train_input.append(input_ids)

        if self.run_type == "train":
            input_ids = []
            labels = []
            for i in range(len(train_input)):
                input_id = train_input[i] + [self.tokenizer.pad_token_id] * (max_input_length - len(train_input[i]))
                label = [-100] * masked_len[i] + input_id[masked_len[i]:]
                input_ids.append(input_id)
                labels.append(label)

            d["inputs"]["input_ids"] = torch.tensor(np.array(input_ids))
            d["inputs"]["labels"] = torch.tensor(np.array(labels))
            d["inputs"]["cluster"] = torch.tensor(np.array(cluster))
        else:
            inputs = self.tokenizer(d["input_context"], padding=True, max_length=self.args.max_seq_length,
                                    truncation=True)
            for k in inputs.keys():
                inputs[k] = torch.tensor(np.array(inputs[k]))
            inputs["cluster"] = torch.tensor(np.array(cluster))
            d["inputs"] = inputs

        return d


def process_raw_data(args, tokenizer, run_type):
    if args.dataset == "sgd":
        data_dir = args.sgd_data_dir
        zero_shot_domain = {"alarm_1", "trains_1", "payment_1", "messaging_1"}
    elif args.dataset == "multiwoz2.1":
        data_dir = args.multiwoz2_1_data_dir
        zero_shot_domain = {"attraction", "hotel", "restaurant", "taxi", "train"}
    else:
        raise ValueError(f"{args.dataset} doesn't exist")

    schema = json.load(open(os.path.join(data_dir, "schema.json"), "r"))

    all_slotnames = []
    for k, v in schema.items():
        all_slotnames.extend(v)

    raw_data = pd.read_json(os.path.join(data_dir, f"{run_type}.json"), encoding="utf-8")
    raw_data = process_valid_dialogue(args, raw_data, zero_shot_domain, run_type)

    print(f"{run_type} raw data: {len(raw_data)}")

    processed_data = get_processed_data(args, raw_data, schema, run_type)

    processed_data = pd.DataFrame(processed_data)
    slotnames = list(set(processed_data["slotname"]))

    cluster_ids = cluster(args, tokenizer, slotnames, all_slotnames, run_type)

    # def add_cluster_id(df):
    #     df["cluster_id"] = cluster_ids[df["slotname"]]
    #     return df

    # processed_data = processed_data.apply(add_cluster_id, axis=1)

    processed_data["cluster_id"] = processed_data["slotname"].map(cluster_ids)

    return processed_data



def data_loader(args, tokenizer, run_type="train"):
    processed_data = process_raw_data(args, tokenizer, run_type)

    if run_type == "train":
        processed_data = sample_data(args, processed_data, run_type)

    if args.num_sample > 0:
        example = processed_data.groupby("slotname")
        example = {k: v for k, v in example}

    if run_type != "test":
        processed_data = processed_data.groupby("cluster_id")
        loader = {}
        for k, v in processed_data:
            v = v.reset_index(drop=True)
            data = DSTState(
                args,
                v,
                example if args.num_sample > 0 else None,
                tokenizer,
                run_type
            )
            loader[k] = DataLoader(
                data,
                batch_size=args.train_batch_size if run_type == "train" else args.evaluate_batch_size,
                collate_fn=data.collate_fn,
                num_workers=6 * (args.num_gpu if "num_gpu" in args else 1),
                shuffle=True if run_type == "train" else False
            )
    else:
        data = DSTState(
            args,
            processed_data,
            example if args.num_sample > 0 else None,
            tokenizer,
            run_type
        )

        loader = DataLoader(
            data,
            batch_size=args.train_batch_size if run_type == "train" else args.evaluate_batch_size,
            collate_fn=data.collate_fn,
            num_workers=6 * (args.num_gpu if "num_gpu" in args else 1),
            shuffle=True if run_type == "train" else False
        )

    print(f"success fully load {run_type} data")
    return loader


def process_valid_dialogue(args, raw_data, zero_shot_domain, run_type):
    def filter(data):
        domains = data["domain"]
        if len(domains) == 0:
            return False
        if args.dataset == "multiwoz2.1":
            # filter hospital and bus
            if len(domains) != len(zero_shot_domain.intersection(set(domains))):
                return False

            if run_type == "train":
                if len(domains) == 1 and args.exclude_domain == domains[0]:
                    return False
                if args.exclude_domain in data["domain"]:
                    data["domain"].remove(args.exclude_domain)
                    assert len(data["domain"]) > 0
            else:
                if args.exclude_domain not in domains:
                    return False
                data["domain"] = [args.exclude_domain]
        elif args.dataset == "sgd":
            if run_type == "test":
                if set(domains).intersection(zero_shot_domain) == 0:
                    return False
                for d in domains:
                    if d not in zero_shot_domain:
                        data["domain"].remove(d)
        else:
            raise ValueError(f"{args.dataset} doesn't exist")
        return True

    raw_data = raw_data[raw_data.apply(filter, axis=1)].reset_index(drop=True)

    return raw_data


def get_processed_data(args, data, schema, run_type):

    processed_data = []
    for idx, d in data.iterrows():
        dialogue_id = d["dialogue_id"]
        dialogue_history = d["dialogue_history"]
        domains = d["domain"]
        possible_slots = []
        for dm in domains:
            possible_slots.extend([(dm, i) for i in schema[dm]])

        assert len(domains) > 0 and len(possible_slots) > 0

        for bf in d["belief_state"]:
            turn = bf["turn"]
            belief_states = {s[1]: s[2] for s in bf["all_states"]}
            utterance = " ".join(dialogue_history[:turn])
            for domain, slot in possible_slots:
                if slot in belief_states:
                    value = belief_states[slot]
                else:
                    value = "none"
                processed_data.append({
                    "dialogue_id": dialogue_id,
                    "slotname": slot,
                    "value": value,
                    "turn": turn,
                    "utterance": utterance,
                    "domain": domain,
                })

    print(f"all processed data {len(processed_data)}")
    return processed_data


def sample_data(args, data, run_type):
    def sample(df: pd.DataFrame):
        none_data = df[df["value"] == "none"]
        valid_data = df[df["value"] != "none"]
        print(f"none data: {len(none_data)} valid data: {len(valid_data)}")
        valid_none_data = none_data.sample(min(len(none_data), int(len(valid_data) * args.none_rate)), random_state=args.random_seed)
        df = pd.concat([valid_data, valid_none_data]).reset_index(drop=True)
        df = df.sort_values(by="dialogue_id").reset_index(drop=True)
        print(f"none: {len(valid_none_data)} labeled: {len(valid_data)}")
        return df

    data = sample(data)

    return data
