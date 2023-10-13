# import asdl
# from asdl import ASDLParser, parse

# from third_party.spider.process_sql import *

# toks = tokenize('select id from w as T1')
# print(scan_alias(toks))
# syntax = """
#     module Lambda {
#         term =
#             Lambda(name x, term body) |
#             Apply(term function, term argument) |
#             Variable(name x)
#     }
# """

# print(ASDLParser().parse(syntax))

# x = asdl.parse("/workspaces/rat-sql/ratsql/grammars/Squall.asdl")

# print(x.sum_type_constructors.keys())
# s = "'\"girl\"'"
# "2008 telstra men's pro"
# '"mister love"'
# s = "(1) \"we will rock you\"\n(2) \"we are the champions\""
# "\"i see dead people\""
# s = 's'
# indexes = []
# single = False
# len_ = len(s)
# i = 0
# start = None
# end = None
# while i<len_:
#     if not single and s[i]=='\'':
#         single = True
#         start = i
#         i += 1
#     if single and s[i]=='\'':
#         end = i
#         single = False
#         indexes.append((start,end))
#         i += 1
#     i += 1
# print(indexes)


import sys, os
import argparse
import json

import _jsonnet
import attr
from ratsql.commands import preprocess, train, infer, eval

@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


exp_config = json.loads(_jsonnet.evaluate_file('/workspaces/rat-sql/configs/spider/nl2code-bert.jsonnet'))
model_config_file = exp_config["model_config"]
if "model_config_args" in exp_config:
    model_config_args = exp_config["model_config_args"]
    if model_config_args is not None:
        model_config_args_json = _jsonnet.evaluate_snippet("", model_config_args)
        model_config_args.update(json.loads(model_config_args_json))
    model_config_args = json.dumps(model_config_args)

eval_config = EvalConfig(
    '/workspaces/rat-sql/configs/spider/nl2code-bert.jsonnet',
    model_config_args,
    "logdir/squall_bert_run",
    "val",
    './',
    './'
)

eval.main(eval_config)