import glob
import json
from utils.logger import run_logger
import os
import pandas as pd
from copy import deepcopy
from utils.data_utils import save_processed_files, unzip_file, save_schema

def process_dialogue(dialogue_id, dialogue):
    all_state = {}
    turn_state = {}
    dialogue_history = []
    utterance = ""
    turn = 1
    domains = set()
    belief_states = []
    for d in dialogue["log"]:
        if len(utterance) == 0:
            utterance = f"user: {d['text']}"
        else:
            utterance = f"{utterance} system: {d['text']}"
            for domain, info in d["metadata"].items():
                for slot, value in info["semi"].items():
                    slotname = f"{domain} {slot}"
                    if len(value) > 0 and value != "not mentioned" and value != "none" and (
                            slotname not in all_state or (value, domain) != all_state[slotname]):
                        turn_state[slotname] = (value, domain)
                        domains.add(domain)
                for slot, value in info["book"].items():
                    if slot != "booked":
                        slotname = f"{domain} {slot}"
                        if len(value) > 0 and value != "not mentioned" and value != "none" and (
                                slotname not in all_state or (value, domain) != all_state[slotname]):
                            turn_state[slotname] = (value, domain)
                            domains.add(domain)
            all_state.update(turn_state)
            belief_states.append({
                "turn": turn,
                "turn_states": [[v[1], k, v[0]] for k, v in turn_state.items()],
                "all_states": [[v[1], k, v[0]] for k, v in all_state.items()],
            })
            turn = turn + 1
            dialogue_history.append(utterance)
            utterance = ""
            turn_state = {}

    data = {
        "dialogue_id": dialogue_id,
        "domain": list(domains),
        "dialogue_history": dialogue_history,
        "belief_state": belief_states
    }
    schema = {}

    for k, v in all_state.items():
        if v[1] not in schema:
            schema[v[1]] = set()
        schema[v[1]].add(k)

    return data, schema


def preprocess1(data_dir, dst_dir):
    data_path = os.path.join(data_dir, "data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path}")

    test_list_name = glob.glob(os.path.join(data_dir, "*testList*"))
    assert len(test_list_name) == 1, "exits multi test list file"
    test_list = open(test_list_name[0], "r").read().split("\n")

    dev_list_name = glob.glob(os.path.join(data_dir, "*valList*"))
    assert len(dev_list_name) == 1, "exits multi val list file"
    dev_list = open(dev_list_name[0], "r").read().split("\n")

    df = pd.read_json(data_path)

    processed_data = {
        "train": [],
        "test": [],
        "dev": []
    }

    all_schema = {}

    for dialogue_id, dialogue in df.items():
        data, schema = process_dialogue(dialogue_id, dialogue)
        if dialogue_id in test_list:
            processed_data["test"].append(data)
        elif dialogue_id in dev_list:
            processed_data["dev"].append(data)
        else:
            processed_data["train"].append(data)
        for k, v in schema.items():
            if k not in all_schema:
                all_schema[k] = set()
            for i in v:
                all_schema[k].add(i)
    save_processed_files(processed_data, dst_dir)
    for k, v in all_schema.items():
        all_schema[k] = list(v)
    save_schema(all_schema, dst_dir)


def process_dialogue2(dialogue):
    all_state_data = []
    single_state_data = []
    dialogue_id = dialogue["dialogue_id"]
    services = dialogue["services"]
    dialogue_history = []
    previous_state = {}
    turn_id = 1

    utterance = ""
    current_state = {}
    domains = set()
    for idx, turn in enumerate(dialogue["turns"]):
        if len(utterance) == 0:
            utterance = f'{turn["speaker"]}: {turn["utterance"]}'
        else:
            utterance = f'{utterance} {turn["speaker"]}: {turn["utterance"]}'

        for frame in turn["frames"]:
            service = frame["service"]
            if "state" in frame.keys() and "slot_values" in frame["state"]:
                state = frame["state"]['slot_values']
                for k, v in state.items():
                    k = k.replace("-", " ")
                    v = v[0]
                    if k not in previous_state or v != previous_state[k]:
                        current_state[k] = (v, service)
                        domains.add(service)
        if turn["speaker"] == "USER":
            if len(current_state) > 0:
                all_state_data.append({
                    "dialogue_id": dialogue_id,
                    "turn": turn_id,
                    "dialogue_history": deepcopy(dialogue_history),
                    "utterance": utterance,
                    "services": list(services),
                    "current_state_slotname": [k for k in current_state.keys()],
                    "current_state_value": [v[0] for v in current_state.values()],
                    "previous_state_slotname": [k for k in previous_state.keys()],
                    "previous_state_value": [v[0] for v in previous_state.values()],
                    "domain": list(domains)
                })

                for k, v in current_state.items():
                    single_state_data.append({
                        "dialogue_id": dialogue_id,
                        "turn": turn_id,
                        "dialogue_history": deepcopy(dialogue_history),
                        "utterance": utterance,
                        "slotname": k,
                        "value": v[0],
                        "domains": list(domains),
                        "domain": v[1]
                    })
            turn_id = turn_id + 1
            previous_state.update(current_state)
            dialogue_history.append(utterance)
            utterance = ""
            current_state = {}
            domains = set()

    return single_state_data, all_state_data


def process_schema1(path):
    schemas = json.load(open(path, "r"))
    all_schema = {}

    for schema in schemas.keys():
        sp = schema.split("-")
        if len(sp) == 2:
            domain, slotname = sp[0], f"{sp[0]} {sp[1]}"
        else:
            if "semi" == sp[1]:
                domain, slotname = sp[0], f"{sp[0]} {sp[2]}"
            else:
                domain, slotname = sp[0], f"{sp[0]} {sp[1]} {sp[2]}"
        if domain in all_schema:
            all_schema[domain].append(slotname)
        else:
            all_schema[domain] = [slotname]

    return all_schema


def process_schema2(path):
    schemas = json.load(open(path, "r"))

    all_schema = {}

    for schema in schemas:
        domain = schema["service_name"]
        slotname = []
        for sn in schema["slots"]:
            slotname.append(sn["name"].replace("-", " "))
        all_schema[domain] = slotname
    return all_schema


def preprocess2(data_dir, dst_dir):
    data_type = ["train", "dev", "test"]

    all_state_data = {
        "train": [],
        "test": [],
        "dev": []
    }
    single_state_data = {
        "train": [],
        "test": [],
        "dev": []
    }

    for dt in data_type:
        files = glob.glob(os.path.join(data_dir, f"{dt}/*.json"))
        for file in files:
            raw_data = pd.read_json(file)
            for _, dialogue in raw_data.iterrows():
                data = process_dialogue2(dialogue)
                single_state_data[dt].extend(data[0])
                all_state_data[dt].extend(data[1])

    all_state_dir = os.path.join(dst_dir, "all_state")
    single_state_dir = os.path.join(dst_dir, "single_state")

    save_processed_files(all_state_data, all_state_dir)
    save_processed_files(single_state_data, single_state_dir)

    schema_path = os.path.join(data_dir, "schema.json")

    schema = process_schema2(schema_path)

    save_schema(schema, dst_dir)


if __name__ == "__main__":
    data_dir = "./sourcedata/multiwoz"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir}")

    run_logger.info(f"source data dir: {data_dir}")
    # unzip_file(os.path.join(data_dir, "data/MultiWOZ_1.0.zip"), os.path.join(data_dir, "data"))
    # unzip_file(os.path.join(data_dir, "data/MultiWOZ_2.0.zip"), os.path.join(data_dir, "data"))
    unzip_file(os.path.join(data_dir, "data/MultiWOZ_2.1.zip"), os.path.join(data_dir, "data"))

    # preprocess1(os.path.join(data_dir, "data/MULTIWOZ2 2"), "./data/multiwoz2.0")
    preprocess1(os.path.join(data_dir, "data/MultiWOZ_2.1"), "./data/multiwoz2.1")
    # preprocess2(os.path.join(data_dir, "data/MultiWOZ_2.2"), "./data/multiwoz2.2")
