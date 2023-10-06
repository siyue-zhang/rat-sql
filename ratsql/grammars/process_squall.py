################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)/present_ref(string)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, val2)
# dual_val: (unit_op, val1, val2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, dual_val1, dual_val2)
# nested_condition: [cond1, 'and'/'or', cond2, ...]
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# query: (query_op, val1, val2)
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': nested_condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
import re

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'notnull', 'isnull')
UNIT_OPS = ('none', '-', '+', "*", '/', 'abs', 'sum')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'julianday', 'length')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

DUAL_VAL_OPS = ('none','-', '+', "*", '/', '>', '<', '>=','<=')
QUERY_OPS = ('none','-', '+', "*", '/', '>', '<', '>=','<=', '!=', '=', 'notnull', 'isnull', 'abs', 'min')

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema



def tokenize(string):
    string = str(string)

    # new code for:
    # "2008 telstra men's pro"
    # '"mister love"'
    # "(1) \"we will rock you\"\n(2) \"we are the champions\""
    # "\"i see dead people\""
    # 'n.d. miss'
    # ( "judge's choice" , 'birth year songs' ) 
    # "'n bietjie"

    tmp = string.replace("\\'","__").replace('\\"',"__")
    patterns = [
        r'\'.*?\d{1,}\/*\d*\".*?\'', # '1/10"'
        r'\"[^\"\']*?\'[^\"\']*?\"' ,
        r'\'[^\"\']*\"[^\"\']{1,}?\"[^\"\']*\'', 
        r'\"[^\"\']*\'[^\"\']{1,}?\'[^\"\']*\"', 
        r'\'[^\"\']{1,}?\'', 
        r'\"[^\"\']{1,}?\"'
        ]
    quote_idxs = []
    for pattern in patterns:
        index = [(m.start(0), m.end(0)) for m in re.finditer(pattern, tmp)]
        # print('find ', index)
        for s, e in index:
            quote_idxs += [s, e-1]
            tmp = tmp[:s] + '_'*(e-s) + tmp[e:]
            # print(tmp)

    quote_idxs.sort()

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]

    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):
    """
        :returns next idx, column id
    """
    idx = start_idx
    tok = toks[idx]
    if debug:
        print('parse col start: ', idx, tok)

    isBlock = False
    if tok=='(':
        isBlock = True
        idx += 1

    if tok == "*":
        idx += 1
        if isBlock:
            assert toks[idx]==")"
            idx += 1
        return idx, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        idx += 1
        if isBlock:
            assert toks[idx]==")"
            idx += 1
        return idx, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            idx += 1
            if isBlock:
                assert toks[idx]==")"
                idx += 1
            return idx, schema.idMap[key]
    
    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    distinct_bracket = False
    agg_id = AGG_OPS.index('none')

    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if debug:
        print('parse col unit start: ', idx, isBlock)
        if idx<len_:
            print('tok: ', toks[idx])

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1

    if toks[idx] == 'distinct':
        isDistinct = True
        idx+=1
        if toks[idx] == '(':
            distinct_bracket = True
            idx+=1

    if debug:
        print('before parse col: ', idx, toks)
        if idx<len_:
            print('tok: ', toks[idx])
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
    
    if debug:
        print('parse col result: ', idx, 'col id: ', col_id)
        if idx<len_:
            print('tok: ', toks[idx])

    if isDistinct and distinct_bracket:
        assert toks[idx] == ')'
        idx += 1

    if agg_id != 0:
        assert toks[idx] == ')'
        idx += 1
    
    if isBlock:
        assert toks[idx] == ')'
        idx += 1 

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):

    idx = start_idx
    len_ = len(toks)
    if debug:
        print('parse val unit start: ', idx, 'total: ', len_)
        if idx<len_:
            print('tok: ', toks[idx])
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # special cases
    isAbs = False
    if toks[idx] == 'abs':
        isAbs = True
        idx += 1
        assert toks[idx] == '('
        idx += 1

    isSum = False
    if idx+3<len_ and toks[idx+3] == '-' and toks[idx] == 'sum':
        isSum = True
        idx += 1
        assert toks[idx] == '('
        idx += 1

    col_unit1 = None
    val2 = None
    unit_op = UNIT_OPS.index('none')
    if debug:
        print('before col unit ', idx, toks[idx])
    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
    if debug:
        print('after col unit', idx, toks[idx])

    if idx < len_ and toks[idx] in UNIT_OPS:
        if toks[idx]=='-' and isAbs:
            unit_op = UNIT_OPS.index('abs')
        elif toks[idx]=='-' and isSum:
            unit_op = UNIT_OPS.index('sum')
        else:
            unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        # idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)

    if isAbs:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    if isSum:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, val2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):
    if debug:
        print('parse value start')
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema, debug=True)
        print('after parse sql: ', idx, 'total: ', len_)
    elif toks[idx] == 'present_ref':
        val = 'present_ref'
        idx += 1
    elif "\"" in toks[idx] or "\'" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            idx, val = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables, debug=debug)

    if debug:
        print('end of parse vale: ', isBlock, idx)
        if idx<len_:
            print('tok', toks[idx])

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_dual_value(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):

    idx = start_idx
    len_ = len(toks)
    # (dual_val_op, val1, val2)
    dual_val_op = 0
    val1 = None
    val2 = None

    if debug:
        print('dual val start ', idx, toks)
        if idx<len_:
            print('tok: ', toks[idx])

    idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables, debug)
    if debug:
        print('first value success: ', idx, 'total: ', len_)
        if idx < len_:
            print(' tok: ', toks[idx])

    if idx<len_ and toks[idx] in DUAL_VAL_OPS:
        dual_val_op = DUAL_VAL_OPS.index(toks[idx])
        idx += 1
        idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables, debug)
    
    return idx, (dual_val_op, val1, val2)


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):
    idx = start_idx
    len_ = len(toks)
    conds = []

    if debug:
        print('parse condition start: idx ', idx, 'token: ', toks[idx], 'total: ', len_, toks )

    isBlock = False # sub condition
    if toks[idx]=='(':
        isBlock = True
        idx+=1

    while idx < len_:
        if debug:
            print('condition run start: idx ', idx, 'token: ', toks[idx])

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
        if debug:
            print('condition run get: ', val_unit)
            print('after run: ', idx, 'total: ', len_)
            if idx<len_:
                print('tok: ', toks[idx])

        not_op = False
        vals = []

        if toks[idx] == 'not' and toks[idx+1] != 'null':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])

        idx += 1
        dual_val1 = dual_val2 = None
        # between case: two values
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, dual_val1 = parse_dual_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, dual_val2 = parse_dual_value(toks, idx, tables_with_alias, schema, default_tables)
        # in case: multiple values
        elif op_id == WHERE_OPS.index('in'):
            if toks[idx] == '(':
                idx += 1
                while toks[idx] != ')':
                    if toks[idx] != ',':
                        idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                        vals.append(val)
                    else:
                        # skip ','
                        idx += 1
                # skip ')'
                idx += 1
            else:
                idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                vals.append(val)
                if toks[idx] == ',':
                    raise ValueError('If "where" condition "in" more than one values, must use "(val1 , val2)" ')
        # is null case
        elif op_id == WHERE_OPS.index('is'):
            if toks[idx]=='null':
                op_id = WHERE_OPS.index('isnull')
                idx += 1
            else:
                raise ValueError('"is" can be only used for "is null" ')
        # not null case
        elif op_id == WHERE_OPS.index('not') and toks[idx]=='null':
                op_id = WHERE_OPS.index('notnull')
                idx += 1
        # normal case: single value
        else:
            if debug:
                print('before parse dual value ', idx, toks[idx], toks)
            idx, dual_val1 = parse_dual_value(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
            dual_val2 = None

        if debug:
            print('after 1 run dual value ', idx, 'total len ', len_)
            if idx<len_:
                print('tok: ', toks[idx])

        conds.append((not_op, op_id, val_unit, dual_val1, dual_val2, vals))
        
        # if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx]in[";",")"] or toks[idx] in JOIN_KEYWORDS):
        #     if isBlock:
        #         idx+=1
        #     break
        
        if idx < len_:
            if toks[idx] in COND_OPS:
                if toks[idx+1]=='(':
                    break
                else:
                    conds.append(toks[idx])
                    idx += 1  # skip and/or
            else:
                if isBlock:
                    idx+=1
                break
        
    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None, debug=False):
    idx = start_idx
    len_ = len(toks)
    if debug:
        print('parse select start: ', idx,  'total len', len_, toks)
    assert toks[idx] == 'select', "'select' not found"
    select_tokens = toks[idx:]
    from_index = select_tokens.index('from')
    select_tokens = select_tokens[:from_index]
    # print('check parse ', select_tokens)

    # max (c1 + c2)
    # count (c1) + count (c2)
    # count ( distinct ( c2 ) )
    if re.search(r'select .* \( .* \) [\+\-] .*', ' '.join(select_tokens)):
        if select_tokens.count('(')>2: # max ( count ( c1 ) + count ( c2 ) )
            agg_inside = False
        else:
            agg_inside = True
    else:
        agg_inside = False

    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []


    agg_bracket = False
    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            if agg_inside:
                pass
            else:
                agg_id = AGG_OPS.index(toks[idx])
                idx += 1
                if toks[idx]=='(':
                    agg_bracket = True
                    idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
        # print('parse_val_unit out ', toks[idx], val_unit)
        if agg_bracket:
            assert toks[idx] == ')'
            agg_bracket = False
            idx += 1
        
        val_units.append((agg_id, val_unit))

        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','



    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_nested_where(toks, start_idx, tables_with_alias, schema, default_tables, debug=False):

    idx = start_idx
    len_ = len(toks)

    if debug:
        print('parse nested where start :', start_idx, 'total: ', len_)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []
    idx += 1
    
    # __: val_unit / op_sql / val
    # where __ and __ and __
    # where ( __ and __ ) and ( __ and __ )
    # where __ and ( __ and __ )

    nested_conds = []
    while idx < len_:
        if debug:
            print('condition run start idx: ', idx, 'token: ', toks[idx], 'total: ', len_)
        
        idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
        nested_conds.append(conds)

        # if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx]in[")",";"] or toks[idx] in JOIN_KEYWORDS):
        #     break
        
        if idx < len_ and toks[idx] in COND_OPS:
            op_id = COND_OPS.index(toks[idx])
            nested_conds.append(op_id)
            idx += 1
        else:
            break
        
    return idx, nested_conds

def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []
    
    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)

    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables, debug=False):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        if debug:
            print('run start order by: ', idx, toks[idx])
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
        if debug:
            print('parse val unit in order by one run :', val_unit)
            print('run next: ', idx)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema, debug=False):
    if debug:
        print('parse sql start: ', start_idx, toks)
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
    idx = from_end_idx
    sql['select'] = select_col_units
    if debug:
        print('parse select success ', select_col_units)
    # where clause
    idx, nested_where_conds = parse_nested_where(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
    # idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = nested_where_conds
    if debug:
        print('nested where success ', nested_where_conds)
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables, debug=debug)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    
    return idx, sql

def parse_sql_query(toks, start_idx, tables_with_alias, schema, nt, debug=False):
    idx = start_idx
    query_op = 0
    val1 = None
    val2 = None
    sql_string = ' '.join(toks).strip()
        
    # Rewrite some SQLs
    # (1) select count ( * ) > 0 from w where c4 not null 
    # --> select ( select count ( * ) from w where c4 not null ) > 0
    check = re.findall(r'^select .*( [>=<]{1,2} \d{1,}) from', sql_string)
    if check:
        print('rewriting1 ', sql_string)
        assert len(check)==1
        check = check[0]
        sql_string = 'select ( ' + sql_string.replace(check, '') + ' )' + check
        toks = tokenize(sql_string)
        print('after rewriting ', sql_string)
    # (2a) select c2_number + c3_number + c4_number from w where c1 = 'kansas'
    # --> select ( select ( c2_number + c3_number ) from w where c1 = 'kansas' ) + ( select c4_number from w where c1 = 'kansas' )
    # (2b) select c2_number + c3_number + 3 from w where c1 = 'kansas'
    # --> select ( select ( c2_number + c3_number ) from w where c1 = 'kansas' ) + 3
    try:
        substring = sql_string[sql_string.index('select'):sql_string.index('from')]
        check = re.findall(r'^select (.{1,}) ([\+\-]) (.{1,}) ([\+\-]) (.{1,})', substring.strip())
        if check:
            print('rewriting2 ', sql_string)
            rest = sql_string.replace(substring,'')
            assert len(check)==1
            c1, s1, c2, s2, c3 = check[0]
            try:
                _ = float(c3)
                sql_string = f'select ( select ( {c1} {s1} {c2} ) {rest} ) {s2} {c3}'
            except Exception:
                sql_string = f'select ( select ( {c1} {s1} {c2} ) {rest} ) {s2} ( select {c3} {rest} )'
            print('after rewriting ', sql_string)
            toks = tokenize(sql_string)
    except Exception:
        pass
    # (3) select min ( min ( c3_number ) , min ( c4_number ) ) from w
    # --> select min ( ( select min ( c3_number ) from w ) , ( select min ( c4_number ) from w ) )
    if nt=='nt-13842':
    # select min ( min ( c3_number ) , min ( c4_number ) ) from w
        print('rewriting3 ', sql_string)
        sql_string = 'select min ( ( select min ( c3_number ) from w ) , ( select min ( c4_number ) from w ) )'
        toks = tokenize(sql_string)
    # (4) missing quotes for string value
    if nt == 'nt-13604':
    # select c3_raw from w where id = ( select id from w where c5 = loss order by id asc limit 1 ) + 1:
        print('rewriting4 ', sql_string)
        sql_string = "select c3_raw from w where id = ( select id from w where c5 = 'loss' order by id asc limit 1 ) + 1"
        toks = tokenize(sql_string)
    # (5) illegal val_unit operator
    if nt == 'nt-12858': 
    # select c4 = c5 from w where c1 = 'svendborg' and c3 = 'goteborgs kvinnliga':
        print('rewriting5 ', sql_string)
        sql_string = "select ( select c4 from w where c1 = 'svendborg' and c3 = 'goteborgs kvinnliga' ) = ( select c5 from w where c1 = 'svendborg' and c3 = 'goteborgs kvinnliga' )"
        toks = tokenize(sql_string)
    # (6) illegal value \"\"
    if nt == 'nt-3262':
    # select count ( c3 ) from w where c8_number != \"\" and c8_number >= 10:
        print('rewriting6 ', sql_string)
        sql_string = "select count ( c3 ) from w where c8_number not null and c8_number >= 10"
        toks = tokenize(sql_string)
    # (7)
    if nt == 'nt-12172':
    # select 159 > ( select c5_number from w where c3_first = 'townsend bell' )
        print('rewriting7 ', sql_string)
        sql_string = "select ( select c5_number from w where c3_first = 'townsend bell' ) < 159"
        toks = tokenize(sql_string)
    # (8)
    if toks[0]=='select' and toks[1]=='present_ref' and toks[3]+toks[4]!='(select':
    # select present_ref - c6_year from w where c2 = 'luandensis'
    # select present_ref - min ( c1_number ) from w
    # NOT: select present_ref - ( select c4_number from w where c1 = 'kalakaua middle school' )
        print('rewriting8 ', sql_string)
        sql_string = f"select present_ref {toks[2]} ( select {' '.join(toks[3:])} )"
        toks = tokenize(sql_string)

    # replace sub query in the bracket by underscore
    toks_template = toks.copy()
    i=0
    while i < len(toks_template):
        if i < len(toks_template)-1 and toks_template[i]=='(' and toks_template[i+1]=='select':
            try:
                i_next, _ = parse_sql(toks, i, tables_with_alias, schema, debug=False)
                i += 1
                while i <= i_next-2:
                    toks_template[i] = '_'*len(toks_template[i])
                    i+=1
                # print('success')
            except Exception:
                # print('fail')
                i+=1
        else:
            i+=1
    toks_template_string = ' '.join(toks_template).strip()
    print(sql_string)
    print(toks_template_string)

    # identify different query structures
    pattern1 = r'(select )(.*)(\( [ _]{1,} \))( .{1,} )(\( [ _]{1,}? \))(.*)'
    pattern2 = r'(select )(\( [ _]{1,} \)) (.{1,})'
    pattern3 = r'select (\d{1,}) ([\+\-]) (\( [ _]{1,} \))'
    search1 = re.findall(pattern1, toks_template_string)
    search2 = re.findall(pattern2, toks_template_string)
    search3 = re.findall(pattern3, toks_template_string)
    exception1 = False
    exception3 = False
    # some exception cases for pattern1
    if search1:
        if len(search1[0][1])>0:
            exception1 = sum([ tok in search1[0][1] for tok in ['abs', 'min']])==0
    if not search3:
        exception3 = re.findall(r'select (present_ref) ([\+\-]) (\( [ _]{1,} \))', toks_template_string)
    # print(search1)
    # print(search2)

    if not exception1 and search1:
        assert len(search1)==1
        print('match 1')
        res = search1[0]
        assert sum([len(x) for x in res]) == len(toks_template_string)
        indexes = []
        pointer = 0
        for r in res:
            indexes.append((pointer, pointer+len(r)))
            pointer += len(r)

        select = sql_string[indexes[0][0]:indexes[0][1]].strip()
        prefix = sql_string[indexes[1][0]:indexes[1][1]].strip()
        sql1 = sql_string[indexes[2][0]:indexes[2][1]].strip()
        query_op = sql_string[indexes[3][0]:indexes[3][1]].strip()
        sql2 = sql_string[indexes[4][0]:indexes[4][1]].strip()
        suffix = sql_string[indexes[5][0]:indexes[5][1]].strip()

        if 'abs' in prefix or 'min' in prefix:
            assert ')' in suffix

        val1 = parse_value(tokenize(sql1), 0, tables_with_alias, schema, debug=debug)
        val2 = parse_value(tokenize(sql2), 0, tables_with_alias, schema, debug=debug)
     
        if query_op=='-' and 'abs' in prefix:
            query_op = QUERY_OPS.index('abs')
        elif query_op==',' and 'min' in prefix:
            query_op = QUERY_OPS.index('min')
        else:
            query_op = QUERY_OPS.index(query_op)

    elif search2:
        assert len(search2)==1
        print('match 2')
        if toks[-1]=='null':
            if toks[-2]=='is':
                query_op = QUERY_OPS.index("isnull")
            elif toks[-2]=='not':
                query_op = QUERY_OPS.index("notnull")
            else:
                NotImplemented
        else:
            query_op = QUERY_OPS.index(toks[-2])
            _, val2 = parse_value(toks, len(toks)-1, tables_with_alias, schema, default_tables=None)
        val1 = parse_value(toks[2:-3], 0, tables_with_alias, schema)

    elif search3 or exception3:
        if exception3:
            search3 = exception3
        assert len(search3)==1
        print('match 3')
        val1 = parse_value(tokenize(toks[1]), 0, tables_with_alias, schema)
        query_op = QUERY_OPS.index(toks[2])
        val2 = parse_value(toks[3:], 0, tables_with_alias, schema)

    elif 'from' not in toks:
        print('match 4')
        # select 2015 - 1912
        assert toks[0]=='select'
        _, val1, query_op, val2 = toks
        val1 = parse_value(tokenize(val1), 0, tables_with_alias, schema)
        query_op = QUERY_OPS.index(query_op)
        val2 = parse_value(tokenize(val2), 0, tables_with_alias, schema)

    else:
        print('match 5')
        _, val1 = parse_value(toks, start_idx, tables_with_alias, schema, debug=debug)
        
    sql_query = (query_op, val1, val2)
    return idx, sql_query

def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query, nt, debug=False):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql_query = parse_sql_query(toks, 0, tables_with_alias, schema, nt, debug=debug)

    # print('Get SQL result:')
    # print('(0) query op: ', QUERY_OPS[sql_query[0]], ', id: ', sql_query[0])
    # print('(1) sql: ', sql_query[1])
    # print('\n(2) val/sql: ', sql_query[2])
    # print('\n')
    return sql_query


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx

