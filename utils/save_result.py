import os.path
import sqlite3
import time


def check_exist_table(cur, table_name):
    res = cur.execute(f"select count(name) from sqlite_master where name like '{table_name}%'")
    if res.fetchone()[0] != 3:
        if table_name == "few_shot":
            create_few_shot_table(cur)
        else:
            create_zero_shot_table(cur)


def create_few_shot_table(cur):
    cur.execute("pragma foreign_keys = on")

    cur.execute("""
    create table if not exists few_shot_result(
    id integer primary key autoincrement,
    uuid integer not null,
    time text not null,
    dataset text not null,
    n_clusters integer not null,
    random_seed integer default 42 not null,
    history_turn integer not null,
    num_sample integer not null,
    max_source_length integer not null,
    max_target_length integer not null,
    pre_seq_len integer not null,
    lr real not null,
    data_ratio integer not null,
    cluster_feature text not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null,
    checkpoint text not null,
    backbone text not null,
    none_rate real not null
    );
    """)

    cur.execute("""
    create table if not exists few_shot_detail_result(
    id integer primary key autoincrement,
    uuid integer not null,
    cluster_id integer not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null
    );
    """)

    cur.execute("""
    create table if not exists few_shot_dev_result(
    id integer primary key autoincrement,
    uuid integer not null,
    step integer not null,
    cluster_id integer not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null
    );
    """)


def create_zero_shot_table(cur):
    cur.execute("pragma foreign_keys = on")

    cur.execute("""
    create table if not exists zero_shot_result(
    id integer primary key autoincrement,
    uuid integer not null,
    time text not null,
    dataset text not null,
    n_clusters integer not null,
    random_seed integer default 42 not null,
    history_turn integer not null,
    num_sample integer not null,
    max_source_length integer not null,
    max_target_length integer not null,
    pre_seq_len integer not null,
    lr real not null,
    domain text not null,
    cluster_feature text not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null,
    checkpoint text not null,
    backbone text not null,
    none_rate real not null
    );
    """)

    cur.execute("""
    create table if not exists zero_shot_detail_result(
    id integer primary key autoincrement,
    uuid integer not null,
    cluster_id integer not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null
    );
    """)

    cur.execute("""
    create table if not exists zero_shot_dev_result(
    id integer primary key autoincrement,
    uuid integer not null,
    step integer not null,
    cluster_id integer not null,
    slot_acc real not null,
    label_acc real not null,
    none_acc real not null,
    JGA real not null
    );
    """)


def insert_few_shot_result(uuid, args, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("few_shot")
    sql = """
    insert into few_shot_result(uuid, time, dataset, n_clusters, random_seed, history_turn, num_sample, 
    max_source_length,
    max_target_length, pre_seq_len, lr, data_ratio, cluster_feature, slot_acc, label_acc, none_acc, JGA, 
    checkpoint, backbone, none_rate)
    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    cur.execute(sql, (uuid, time.strftime("%y-%m-%d %H:%M:%S", time.localtime()), args.dataset, args.n_clusters,
                      args.random_seed, args.history_turn, args.num_sample, args.max_source_length,
                      args.max_target_length, args.pre_seq_len, args.lr, args.data_ratio, args.cluster_feature,
                      slot_acc, label_acc, none_acc, JGA, args.checkpoint, args.backbone, args.none_rate))
    close_db(con, cur)


def insert_few_shot_detail_result(uuid, cluster_id, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("few_shot")
    sql = """
    insert into few_shot_detail_result(uuid, cluster_id, slot_acc, label_acc, none_acc, JGA)
    values(?, ?, ?, ?, ?, ?);
    """
    cur.execute(sql, (uuid, cluster_id, slot_acc, label_acc, none_acc, JGA))
    close_db(con, cur)


def insert_few_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("few_shot")
    sql = """
    insert into few_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA)
    values(?, ?, ?, ?, ?, ?, ?)
    """
    cur.execute(sql, (uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA))
    close_db(con, cur)


def insert_zero_shot_result(uuid, args, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("zero_shot")
    sql = """
    insert into zero_shot_result(uuid, time, dataset, n_clusters, random_seed, history_turn, num_sample, 
    max_source_length, max_target_length, pre_seq_len, lr, domain, cluster_feature, slot_acc, label_acc, none_acc, JGA, 
    checkpoint, backbone, none_rate)
    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    cur.execute(sql, (uuid, time.strftime("%y-%m-%d %H:%M:%S", time.localtime()), args.dataset, args.n_clusters,
                      args.random_seed, args.history_turn, args.num_sample, args.max_source_length,
                      args.max_target_length, args.pre_seq_len, args.lr, args.exclude_domain, args.cluster_feature,
                      slot_acc, label_acc, none_acc, JGA, args.checkpoint, args.backbone, args.none_rate))
    close_db(con, cur)


def insert_zero_shot_detail_result(uuid, cluster_id, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("zero_shot")
    sql = """
    insert into zero_shot_detail_result(uuid, cluster_id, slot_acc, label_acc, none_acc, JGA)
    values(?, ?, ?, ?, ?, ?);
    """
    cur.execute(sql, (uuid, cluster_id, slot_acc, label_acc, none_acc, JGA))
    close_db(con, cur)


def insert_zero_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA):
    con, cur = connect_db("zero_shot")
    sql = """
    insert into zero_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA)
    values(?, ?, ?, ?, ?, ?, ?)
    """
    cur.execute(sql, (uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA))
    close_db(con, cur)


def connect_db(table_name):
    if not os.path.exists("./result"):
        os.makedirs("./result")
    con = sqlite3.connect("./result/record.db")
    cur = con.cursor()
    check_exist_table(cur, table_name)
    return con, cur


def close_db(con, cur):
    con.commit()
    cur.close()
    con.close()
