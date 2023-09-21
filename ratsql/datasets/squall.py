import json, re
from tqdm import tqdm

import sys, os
print(os.getcwd)
sys.path.append("./third_party/spider/preprocess")
sys.path.append("./third_party/wikisql/")

import networkx as nx
import numpy as np
import torch

from ratsql.datasets import spider
from ratsql.utils import registry

# from third_party.squall.model.evaluator import Evaluator
from datasets import load_dataset
from ratsql.datasets.squall_lib.utils import normalize
from third_party.spider.process_sql import *
from third_party.spider.preprocess.parse_sql_one import Schema as Schema_spider

def preprocess_datasets(subset_name, save_dir, limit=None):
    subset_name = str(subset_name)
    raw_dataset = load_dataset("siyue/squall", subset_name)
    for split in ["train", "validation", "test"]:
        dataset = raw_dataset[split]
        num_sample = len(dataset["nt"])
        examples = {}
        schema_dicts = {}

        for i in tqdm(range(num_sample)):
            tbl = dataset["tbl"][i]
            db_id = tbl
            if tbl not in schema_dicts:
                table_json = f"/workspaces/rat-sql/data/squall/tables/json/{tbl}.json"
                f = open(table_json)
                table_json = json.load(f)
                header = table_json["headers"][2:]
                header_clean = [normalize(h)  for h in header]
                column_names = []
                types = []
                contents = table_json["contents"][2:]
                for c in contents:
                    for cc in c:
                        types.append(cc["type"])
                        match = re.search(r'c(\d+)', cc["col"])
                        number_after_c = int(match.group(1))
                        col_name = re.sub(r'c(\d+)', '{}'.format(header_clean[number_after_c-1]), cc["col"])
                        column_names.append(col_name)
                types_rev = []
                for t in types:
                    if t in ["INTEGER", "LIST INTEGER", "REAL","LIST REAL"]:
                        types_rev.append("real")
                    else:
                        types_rev.append("text")

                schema_dicts[db_id] = {
                    "tbl": tbl, 
                    "header":header_clean,
                    "columns":column_names,
                    "column_types":types_rev}
            
            header = schema_dicts[db_id]["header"]
            sql = dataset["sql"][i]
            sql_rev = []
            for token in sql["value"]:
                match = re.search(r'c(\d+)', token)
                if match:
                    number_after_c = int(match.group(1))
                    token_rev = re.sub(r'c(\d+)', '{}'.format(header[number_after_c-1]), token)
                    sql_rev.append(token_rev)
                elif token == 'w':
                    sql_rev.append(tbl)
                else:
                    sql_rev.append(token)
            sql_rev = ' '.join(sql_rev)
            examples[db_id] = {
                "nt": dataset["nt"][i],
                "question_toks":dataset["nl"][i],
                "sql": sql_rev,
                "tables": [tbl],
                "columns": schema_dicts[db_id]["columns"],
                "column_types":schema_dicts[db_id]["column_types"],
                "header": schema_dicts[db_id]["header"],
                "tgt": dataset["tgt"][i]
            }
            if limit and i>limit:
                break

        with open(f"{save_dir}/{split}{subset_name}.json", "w") as file:
            json.dump(examples, file)

        print(f"Save squall {split} set!")

    return

@registry.register('dataset', 'squall')
class SquallDataset(torch.utils.data.Dataset): 
    def __init__(self, path, limit=None):
        f = open(path)
        data = json.load(f)
        self.examples = []
        for k in data:
            # only 1 table in squall
            example = data[k]
            tbl = example["tables"][0]
            tables = (spider.Table(
                id=0,
                name=[tbl],
                unsplit_name=tbl,
                orig_name=tbl,
            ),)
            column_names = example["columns"]
            column_types = example["column_types"]
            columns = tuple(
                spider.Column(
                    id=i,
                    table=tables[0],
                    name=col_name.split(),
                    unsplit_name=col_name,
                    orig_name=col_name,
                    type=col_type,
                )
                for i, (col_name, col_type) in enumerate(zip(
                    column_names,
                    column_types
                ))
            )
            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column) 
            # No primary keys
            # No foreign keys
            foreign_key_graph = nx.DiGraph()
            # Final argument: don't keep the original schema
            schema = spider.Schema(tbl, tables, columns, foreign_key_graph, None)
            
            print(example["sql"])
            # sql = "SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
            sql = example["sql"]
            table_spider = {'table_names_original':tbl}
            table_spider['column_names_original'] = [[0, c] for c in column_names]
            schema_spider = Schema_spider({tbl:column_names}, table_spider)
            print({tbl:column_names},'\n')
            print(table_spider)
            sql_label = get_sql(schema_spider, sql)
            print(sql_label)
            assert 1==2

            item = spider.SpiderItem(
                text=example["question_toks"],
                code=sql_label,
                schema=schema,
                orig={
                    'question': example["question_toks"],
                },
                orig_schema=None)
            
            self.examples.append(item)

            if limit and len(self.examples) >= limit:
                break

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
    # preprocess_datasets(subset_name=1, save_dir="/workspaces/rat-sql/data/squall", limit=10)
    # a = SquallDataset(path="/workspaces/rat-sql/data/squall/train1.json", limit=2)
    print(tokenize("sh she_is"))
    # print(a.__getitem__(0))