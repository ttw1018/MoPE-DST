import json
import glob
import os
from os.path import join
from utils.data_utils import save_schema
import pandas as pd


def read_json(file_name):
    """
    read train(test/dev) file
    """
    try:
        fin = open(file_name, "r")
    except Exception as e:
        raise e

    fin = json.load(fin)
    return fin


def save_processed_file(data, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    for k, v in data.items():
        json.dump(v, open(os.path.join(dir, f"{k}.json"), "w"), indent=2, ensure_ascii=False)


def process_dialogue(dialogue):
    dialogue_id = dialogue["dialogue_id"]
    dialogue_history = []
    turn_id = 1
    belief_state = []
    all_states = {}
    turn_states = {}
    utterance = ""
    domains = set()

    for idx, turn in enumerate(dialogue["turns"]):
        if len(utterance) == 0:
            utterance = f'{turn["speaker"]}: {turn["utterance"]}'
        else:
            utterance = f'{utterance} {turn["speaker"]}: {turn["utterance"]}'

        if turn["speaker"] == "USER":
            for frame in turn["frames"]:
                service = frame["service"].lower()
                if "state" in frame.keys() and "slot_values" in frame["state"]:
                    state = frame["state"]['slot_values']
                    for k, v in state.items():
                        slotname = f"{service.split('_')[0]} {' '.join(k.split('_'))}"
                        value = v[0]
                        for _v in v:
                            if _v in utterance:
                                value = _v
                        if len(value) > 0 and value.lower() != "none" and (slotname not in all_states
                                                                           or (value, service) != all_states[slotname]):
                            all_states[slotname] = (value, service)
                            turn_states[slotname] = (value, service)
                            domains.add(service)
            belief_state.append({
                "turn": turn_id,
                "turn_states": [[v[1], k, v[0]] for k, v in turn_states.items()],
                "all_states": [[v[1], k, v[0]] for k, v in all_states.items()]
            })
            turn_id = turn_id + 1
            dialogue_history.append(utterance)
            utterance = ""
            turn_states = {}
    data = {
        "dialogue_id": dialogue_id,
        "domain": list(domains),
        "dialogue_history": dialogue_history,
        "belief_state": belief_state
    }

    schema = {}
    for k, v in all_states.items():
        if v[1] not in schema:
            schema[v[1]] = set()
        schema[v[1]].add(k)
    return data, schema


def list_dir_file(dir):
    """
    list all the json file in the directory
    """
    files = glob.glob(os.path.join(dir, "dialogue*.json"))
    return files


def process(data_dir, dst_dir):
    dir_type = ["train", "test", "dev"]
    processed_data = {
        "train": [],
        "test": [],
        "dev": []
    }
    all_schema = {}
    for i in dir_type:
        files = list_dir_file(join(data_dir, i))
        for file in files:
            dialogues = read_json(file)
            for dialogue in dialogues:
                data, schema = process_dialogue(dialogue)
                processed_data[i].append(data)
                for k, v in schema.items():
                    if k not in all_schema:
                        all_schema[k] = set()
                    for _v in v:
                        all_schema[k].add(_v)

    save_processed_file(processed_data, dst_dir)
    for k, v in all_schema.items():
        all_schema[k] = list(v)
    save_schema(all_schema, dst_dir)


def analysis_schema():
    # dir_type = ["train", "test", "dev"]
    dir_type = ["train"]
    slot_names = set()
    for i in dir_type:
        file_name = f"sgd/{i}/schema.json"
        schemas = json.load(open(file_name, "r"))
        print(len(schemas))
        for schema in schemas:
            service_name = schema["service_name"].split("_")[0]
            for slot in schema["slots"]:
                name = f'{service_name}-{slot["name"]}'
                if name in slot_names:
                    print(name)
                else:
                    slot_names.add(name)
    print(len(slot_names))


def get_schema(data_dir, save_dir):
    schemas = {}

    for data_type in ["train", "test", "dev"]:

        schema_file_name = join(join(data_dir, data_type), "schema.json")
        if not os.path.exists(schema_file_name):
            raise FileNotFoundError(schema_file_name)

        df = pd.read_json(schema_file_name)

        for k, v in df.iterrows():
            domain = v["service_name"]
            for slot in v["slots"]:
                slotname = slot["name"]
                if domain not in schemas:
                    schemas[domain] = set()
                schemas[domain].add(f"{domain} {slotname}")

    schemas = {
        k: list(v) for k, v in schemas.items()
    }
    save_schema(schemas, save_dir)

    return schemas


if __name__ == "__main__":
    data_dir = "./sourcedata/dstc8-schema-guided-dialogue/"

    dst_dir = "./data/sgd"
    process(data_dir, dst_dir)
