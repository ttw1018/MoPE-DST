import os
import zipfile
import json
from utils.logger import run_logger


def unzip_file(zip_src, dst_dir):
    if not os.path.exists(zip_src):
        raise FileExistsError(f"{zip_src} not exist")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if zipfile.is_zipfile(zip_src):
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        raise FileExistsError(f"{zip_src} is not zip file")


def save_processed_files(data, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for k, v in data.items():
        run_logger.info(f"save file {dst_dir}/{k}.json")
        json.dump(v, open(f"{dst_dir}/{k}.json", "w"), indent=2, ensure_ascii=False)


def save_schema(data, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    run_logger.info(f"save file {dst_dir}/schema.json")
    json.dump(data, open(f"{dst_dir}/schema.json", "w"), indent=2, ensure_ascii=False)
