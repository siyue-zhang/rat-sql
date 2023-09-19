import json, re

import networkx as nx
import numpy as np
import torch

from ratsql.datasets import spider
from ratsql.utils import registry

# from third_party.squall.model.evaluator import Evaluator
from datasets import load_dataset
from ratsql.datasets.squall_lib.utils import normalize

# raw_datasets = load_dataset(data_args.dataset_name, data_args.subset_name)


@registry.register('dataset', 'squall')
class SquallDataset(torch.utils.data.Dataset): 
    def __init__(self, subset_name, split_name):
        self.subset_name = subset_name
        self.split_name = split_name
        self.raw_dataset = load_dataset("siyue/squall", subset_name)[self.split_name]
        self.examples = []
        self.schema_dicts = {}

        for example in self.raw_dataset:
            tbl = example["tbl"]
            # Only one table in SQUALL (without a real name)
            db_id = tbl
            if tbl not in self.schema_dicts:
                tables = (spider.Table(
                    id=0,
                    name=[db_id],
                    unsplit_name=db_id,
                    orig_name=db_id,
                ),)

                table_json = f"/workspaces/rat-sql/data/squall/tables/json/{tbl}.json"
                f = open(table_json)
                table_json = json.load(f)
                header = table_json["headers"]
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
                columns = tuple(
                    spider.Column(
                        id=i,
                        table=tables[0],
                        name=col_name.split(),
                        unsplit_name=col_name,
                        orig_name=orig_col_name,
                        type=col_type,
                    )
                    for i, (col_name, orig_col_name, col_type) in enumerate(zip(
                        column_names,
                        column_names,
                        types
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
                self.schema_dicts[db_id] = spider.Schema(db_id, tables, columns, foreign_key_graph, None)
            break

        print(self.schema_dicts)

        # for path in paths:
        #     for line in open(path):
        #         entry = json.loads(line)
        #         item = spider.SpiderItem(
        #             text=entry['question'],
        #             code=entry['sql'],
        #             schema=self.schema_dicts[entry['table_id']],
        #             orig={
        #                 'question': entry['question'],
        #             },
        #             orig_schema=None)
        #         self.examples.append(item)

        #         if limit and len(self.examples) > limit:
        #             return

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return self.raw_dataset[idx]

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

    #         return {
    #             'per_item': [{'ex': ex, 'lf': lf} for ex, lf in zip(self.exec_match, self.lf_match)],
    #             'total_scores': {'ex': mean_exec_match, 'lf': mean_lf_match},
    #         }


if __name__=="__main__":
    a = SquallDataset(subset_name="1", split_name="train")
