import json, re
from tqdm import tqdm

import sys, os
sys.path.append("./third_party/spider/preprocess")
sys.path.append("./third_party/spider")
sys.path.append("./third_party/wikisql/")

import networkx as nx
import numpy as np
import torch
from pathlib import Path
import sqlite3

from ratsql.datasets import spider
from ratsql.utils import registry

# from third_party.squall.model.evaluator import Evaluator
from datasets import load_dataset
from ratsql.datasets.squall_lib.utils import normalize
from third_party.spider.process_sql import *
from ratsql.grammars.process_squall import get_sql as get_sql_squall
from third_party.spider.preprocess.parse_sql_one import Schema as Schema_spider

def preprocess_datasets(subset_name, save_dir, limit=None):
    subset_name = str(subset_name)
    raw_dataset = load_dataset("siyue/squall", subset_name)
    for split in ["train", "validation", "test"]:
        dataset = raw_dataset[split]
        num_sample = len(dataset["nt"])
        examples = []
        schema_dicts = {}

        for i in tqdm(range(num_sample)):
            tbl = dataset["tbl"][i]
            db_id = tbl
            if tbl not in schema_dicts:
                table_json = f"/workspaces/rat-sql/data/squall/tables/json/{tbl}.json"
                f = open(table_json)
                table_json = json.load(f)
                header = table_json["headers"]
                header = [h.replace(" ", '_') for h in header]
                header_clean = [normalize(h)  for h in header]
                column_names = []
                orig_column_names = []
                types = []
                contents = table_json["contents"]
                for c in contents:
                    for cc in c:
                        types.append(cc["type"])
                        orig_column_names.append(cc["col"])
                        match = re.search(r'c(\d+)', cc["col"])
                        if match:
                            number_after_c = int(match.group(1))
                            col_name = re.sub(r'c(\d+)', '{}'.format(header_clean[number_after_c-1]), cc["col"])
                            column_names.append(col_name)
                        else:
                            column_names.append(cc["col"])
                types_rev = []
                for t in types:
                    if t in ["INTEGER", "LIST INTEGER", "REAL","LIST REAL"]:
                        types_rev.append("number")
                    else:
                        types_rev.append("text")

                schema_dicts[db_id] = {
                    "tbl": tbl, 
                    "header":header_clean,
                    "columns":column_names,
                    "column_types":types_rev,
                    "orig_column_names":orig_column_names}
            
            header = schema_dicts[db_id]["header"]
            sql = dataset["sql"][i]
            examples.append({
                "nt": dataset["nt"][i],
                "question":' '.join(dataset["nl"][i]),
                "sql": ' '.join(sql["value"]),
                "tables": [tbl],
                "columns": schema_dicts[db_id]["columns"],
                "column_types":schema_dicts[db_id]["column_types"],
                "orig_columns": schema_dicts[db_id]["orig_column_names"],
                "header": schema_dicts[db_id]["header"],
                "tgt": dataset["tgt"][i]
            })
            if limit and i>limit:
                break

        with open(f"{save_dir}/{split}{subset_name}.json", "w") as file:
            json.dump(examples, file)

        print(f"Save squall {split} set! {save_dir}/{split}{subset_name}.json")

    return

# @registry.register('dataset', 'squall')
class SquallDataset(torch.utils.data.Dataset): 
    def __init__(self, path, db_path, limit=None, save_json=False):
        self.raw_examples = json.load(open(path))
        self.examples = []
        self.total_num_examples = 0
        count = 0
        self.save = {}
        for jj, example in enumerate(self.raw_examples):
            if example['nt'] in ['nt-6989', 'nt-4316']:
                # skip these questions which have only 1 case but case complex structure change
                
                # nt-6989
                # select ( select ( ______ ) = ( ______ ) ) and ( select ( ______ ) = ( ______ ) )
                
                # nt-4316
                # select c4 from w group by c4 order by count ( c5_number1 > c5_number2 ) desc limit 1
                continue
            count += 1
            if example['nt'] != 'n':
            # if example['nt'] == 'nt-13348':
                # only 1 table in squall
                self.total_num_examples += 1
                tbl = example["tables"][0]
                tables = (spider.Table(
                    id=0,
                    name=['w'],
                    unsplit_name='w',
                    orig_name=tbl,
                ),)
                column_names = example["columns"]
                column_types = example["column_types"]
                orig_column_names = example["orig_columns"]
                columns = tuple(
                    spider.Column(
                        id=i+1,
                        table=tables[0],
                        name=col_name.replace("_", " ").split(),
                        unsplit_name=col_name.replace("_", " "),
                        orig_name=orig_col_name,
                        type=col_type,
                    )
                    for i, (col_name, col_type, orig_col_name) in enumerate(zip(
                        column_names,
                        column_types,
                        orig_column_names
                    ))
                )
                zero_column = (spider.Column(
                    id = 0,
                    table = tables[0],
                    name = ['*'],
                    unsplit_name = '*',
                    orig_name = '*',
                    type = 'text',
                ),)
                columns = zero_column + columns
                # Link columns to tables
                for column in columns:
                    if column.table:
                        column.table.columns.append(column) 
                # No primary keys
                # No foreign keys
                foreign_key_graph = nx.DiGraph()
                # Final argument: don't keep the original schema
                schema = spider.Schema(tbl, tables, columns, foreign_key_graph, None)
                
                # convert the sql query format for the grammar
                sql = example["sql"]
                print(sql)
                nt = example['nt']
                table_spider = {'table_names_original':['w']}
                table_spider['column_names_original'] = [[0, c] for c in orig_column_names]
                table_spider['column_names_original'] = [[-1, '*']] + table_spider['column_names_original']
                schema_spider = Schema_spider({'w':orig_column_names}, table_spider)
                parsable = False
                # sql = "select ( select c2 from w where c1_number = 2 ) not null"
                # select c1 from w where c2 = '"girl"' intersect select c1 from w where c2 = '"e-pro"'
                # if sql == "select count ( distinct ( c2 ) ) from w where c7 = 'win' and abs ( c6_number1 - c6_number2 ) > 3":
                print('\n', jj, ' ', nt)
                # print(table_spider['column_names_original'])
                # if 'present_ref' in sql:
                #     print(f'found wired keyword present_ref in {sql}')
                #     continue
                # if 'julianday' in sql:
                #     print(f'found wired keyword julianday in {sql}')
                #     continue
                # if 'length (' in sql:
                #     print(f'found wired keyword julianday in {sql}')
                #     continue
                sql_label = get_sql_squall(schema_spider, sql, nt, debug=True)  # Attempt to run get_sql()
                self.save[nt] = {'sql': sql, 'parsed': sql_label}
                # assert 1==2
                # try:
                #     sql_label = get_sql(schema_spider, sql)  # Attempt to run get_sql()
                #     parsable = True  # If get_sql() runs without error, set parsable to True
                # except Exception as e:
                #     # print(f"\nSQL can not be parsed ({example['nt']})! \n {sql}")
                #     pass

                if parsable:
                    col_names = [[0, '*'],]
                    for idx, name in enumerate(column_names):
                        col_names.append([idx+1, name.replace("_", " ")])
                    table_spider['column_names'] = col_names
                    table_spider['table_names'] = ['w']
                    table_spider['db_id'] = tbl
                    table_spider['foreign_keys'] = None
                    table_spider['primary_keys'] = None

                    item = spider.SpiderItem(
                        text=example["question"],
                        code=sql_label,
                        schema=schema,
                        orig={
                            'question': example["question"],
                        },
                        orig_schema=table_spider)
                    
                    sqlite_path = f"{db_path}/{tbl}.db"
                    source: sqlite3.Connection
                    with sqlite3.connect(str(sqlite_path)) as source:
                        dest = sqlite3.connect(':memory:')
                        dest.row_factory = sqlite3.Row
                        source.backup(dest)
                    item.schema.connection = dest

                    self.examples.append(item)

                if limit and len(self.total_num_examples) >= limit:
                    break

                # print(item.schema.columns[0].orig_name)
                # assert 1==2

        if save_json:
            with open('./data/squall/rat-sql/parsed_squall.json', "w") as json_file:
                json.dump(self.save, json_file, indent=4)
        print("total: ", self.total_num_examples, "parsable: ", len(self.examples))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    # class Metrics:
    #     def __init__(self, dataset):
    #       self.dataset = dataset
    #       self.db_engine = dbengine.DBEngine(dataset.db_path)

    #       self.lf_match = []
    #       self.exec_match = []

    #     def _evaluate_one(self, item, inferred_code):
    #         gold_query = query.Query.from_dict(item.code, ordered=False)
    #         gold_result = self.db_engine.execute_query(item.schema.db_id, gold_query, lower=True)

    #         pred_query = None
    #         pred_result = None
    #         try:
    #             pred_query = query.Query.from_dict(inferred_code, ordered=False)
    #             pred_result = self.db_engine.execute_query(item.schema.db_id, pred_query, lower=True)
    #         except:
    #             # TODO: Use a less broad exception specifier
    #             pass

    #         lf_match = gold_query == pred_query
    #         exec_match = gold_result == pred_result

    #         return lf_match, exec_match

    #     def add(self, item, inferred_code, orig_question=None):
    #         lf_match, exec_match = self._evaluate_one(item, inferred_code)
    #         self.lf_match.append(lf_match)
    #         self.exec_match.append(exec_match)

    #     def finalize(self):
    #         mean_exec_match = float(np.mean(self.exec_match))
    #         mean_lf_match = float(np.mean(self.lf_match))

            # return {
            #     'per_item': [{'ex': ex, 'lf': lf} for ex, lf in zip(self.exec_match, self.lf_match)],
            #     'total_scores': {'ex': mean_exec_match, 'lf': mean_lf_match},
            # }


if __name__=="__main__":
    # preprocess_datasets(subset_name=1, save_dir="/workspaces/rat-sql/data/squall/rat-sql")
    a = SquallDataset(path="/workspaces/rat-sql/data/squall/rat-sql/train1.json", db_path='/workspaces/rat-sql/data/squall/tables/db', save_json=True)
    # print(a.__getitem__(0).schema.tables[0].columns[0])
    # print("\n")
    # print(a.__getitem__(0).schema.tables[0].columns[1])
    