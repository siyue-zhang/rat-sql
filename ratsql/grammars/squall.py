import collections
import copy
import itertools
import os

import asdl
import attr
import networkx as nx

from ratsql import ast_util
from ratsql.utils import registry


def bimap(first, second):
    return {f: s for f, s in zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


def join(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def intersperse(delimiter, seq):
    return itertools.islice(
        itertools.chain.from_iterable(
            zip(itertools.repeat(delimiter), seq)), 1, None)


@registry.register('grammar', 'squall')
class SquallLanguage:
    root_type = 'sql'

    def __init__(
            self,
            output_from=False,
            use_table_pointer=False,
            include_literals=True,
            include_columns=True,
            end_with_from=False,
            clause_order=None,
            infer_from_conditions=False,
            factorize_sketch=0):

        # collect pointers and checkers
        custom_primitive_type_checkers = {}
        self.pointers = set()
        if use_table_pointer:
            custom_primitive_type_checkers['table'] = lambda x: isinstance(x, int)
            self.pointers.add('table')
        self.include_columns = include_columns
        if include_columns:
            custom_primitive_type_checkers['column'] = lambda x: isinstance(x, int)
            self.pointers.add('column')

        # create ast wrapper
        asdl_file = "Squall.asdl"
        self.ast_wrapper = ast_util.ASTWrapper(
            asdl.parse(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    asdl_file)),
            custom_primitive_type_checkers=custom_primitive_type_checkers)
        if not use_table_pointer:
            self.ast_wrapper.singular_types['Table'].fields[0].type = 'int'
        if not include_columns:
            col_unit_fields = self.ast_wrapper.singular_types['col_unit'].fields
            assert col_unit_fields[1].name == 'col_id'
            del col_unit_fields[1]

        # literals of limit field
        self.include_literals = include_literals
        if not self.include_literals:
            limit_field = self.ast_wrapper.singular_types['sql'].fields[6]
            assert limit_field.name == 'limit'
            limit_field.opt = False
            limit_field.type = 'singleton'

        # from field
        self.output_from = output_from
        self.end_with_from = end_with_from
        self.clause_order = clause_order
        self.infer_from_conditions = infer_from_conditions
        if self.clause_order:
            # clause order is prioritized over configurations like end_with_from
            assert factorize_sketch == 2  # TODO support other grammars
            sql_fields = self.ast_wrapper.product_types['sql'].fields
            letter2field = {k: v for k, v in zip("SFWGOI", sql_fields)}
            new_sql_fields = [letter2field[k] for k in self.clause_order]
            self.ast_wrapper.product_types['sql'].fields = new_sql_fields
        else:
            if not self.output_from:
                sql_fields = self.ast_wrapper.product_types['sql'].fields
                assert sql_fields[1].name == 'from'
                del sql_fields[1]
            else:
                sql_fields = self.ast_wrapper.product_types['sql'].fields
                assert sql_fields[1].name == "from"
                if self.end_with_from:
                    sql_fields.append(sql_fields[1])
                    del sql_fields[1]

    def parse(self, code, section):
        print('code to parse: ', code)
        return self.parse_sql_query(code)

    def unparse(self, tree, item):
        unparser = SquallUnparser(self.ast_wrapper, item.schema, self.factorize_sketch)
        return unparser.unparse_sql(tree)

    @classmethod
    def tokenize_field_value(cls, field_value):
        if isinstance(field_value, bytes):
            field_value_str = field_value.encode('latin1')
        elif isinstance(field_value, str):
            field_value_str = field_value
        else:
            field_value_str = str(field_value)
            if field_value_str[0] == '"' and field_value_str[-1] == '"':
                field_value_str = field_value_str[1:-1]
        # TODO: Get rid of surrounding quotes
        return [field_value_str]

    def parse_val(self, val):
        print('current val: ', val)
        if isinstance(val, str):
            if not self.include_literals:
                return {'_type': 'Terminal'}
            elif val=='present_ref':
                return {
                    '_type': 'Present_ref',
                    's': val,
                }
            else:
                return {
                    '_type': 'String',
                    's': val,
                }
        elif isinstance(val, list) or isinstance(val, tuple):
            return {
                '_type': 'ColUnit',
                'c': self.parse_col_unit(val),
            }
        elif isinstance(val, float):
            if not self.include_literals:
                return {'_type': 'Terminal'}
            return {
                '_type': 'Number',
                'f': val,
            }
        elif isinstance(val, dict):
            return {
                '_type': 'ValSql',
                's': self.parse_sql(val),
            }
        else:
            raise ValueError(val)

    def parse_col_unit(self, col_unit):
        agg_id, col_id, is_distinct = col_unit
        result = {
            '_type': 'col_unit',
            'agg_id': {'_type': self.AGG_TYPES_F[agg_id]},
            'is_distinct': is_distinct,
        }
        if self.include_columns:
            result['col_id'] = col_id
        return result

    def parse_val_unit(self, val_unit):
        unit_op, col_unit1, val1 = val_unit
        result = {
            '_type': self.UNIT_TYPES_F[unit_op],
            'col_unit1': self.parse_col_unit(col_unit1),
        }
        if unit_op != 0:
            result['val1'] = self.parse_val(val1)
        return result

    def parse_table_unit(self, table_unit):
        table_type, value = table_unit
        if table_type == 'sql':
            return {
                '_type': 'TableUnitSql',
                's': self.parse_sql(value),
            }
        elif table_type == 'table_unit':
            return {
                '_type': 'Table',
                'table_id': value,
            }
        else:
            raise ValueError(table_type)

    def parse_cond(self, cond):
        if cond==None or len(cond)==0:
            return None

        if len(cond) > 1:
            return {
                '_type': self.LOGIC_OPERATORS_F[cond[1]],
                'left': self.parse_cond(cond[:1]),
                'right': self.parse_cond(cond[2:]),
            }

        (not_op, op_id, val_unit, dual_val1, dual_val2, vals), = cond
        # 'Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like', 'Is', 'Notnull', 'Isnull'
        _type = self.COND_TYPES_F[op_id]
        result = {
            '_type': _type,
            'val_unit': self.parse_val_unit(val_unit),}
        if _type == 'Between':
            result['dual_val1'] = self.parse_dual_val(dual_val1)
            result['dual_val2'] = self.parse_dual_val(dual_val2)
        elif _type == 'In':
            result['vals'] = [self.parse_val(v) for v in vals]
        elif _type in ['Notnull', 'Isnull']:
            pass
        else:
            result['dual_val1'] = self.parse_dual_val(dual_val1)
            
        if not_op:
            result = {
                '_type': 'Not',
                'c': result,
            }
        return result


    def parse_dual_val(self, dual_val):
        unit_op, val1, val2 = dual_val
        result = {
            '_type': self.DUAL_VAL_OPERATORS_F[unit_op],
            'val1': self.parse_val(val1),
        }
        if unit_op==0:
            assert val2==None
        if val2:
            result['val2'] = self.parse_val(val2)
        return result

    def parse_nested_cond(self, nested_conds):
        print('nested conds: ', nested_conds, len(nested_conds))
        if len(nested_conds)==0:
            return None
        
        if len(nested_conds)>1:
            return {
                '_type': self.LOGIC_OPERATORS_NEST_F[nested_conds[1]],
                'nleft': self.parse_nested_cond(nested_conds[:1]),
                'nright': self.parse_nested_cond(nested_conds[2:]),
            }
        
        print('nested_conds:', nested_conds)
        conds = nested_conds[0]
        result = self.parse_cond(conds)
        print('parse cond result: ', result)
        return result


    def parse_sql(self, sql):
        if sql is None:
            return None
        
        return filter_nones({
            '_type': 'sql',
            'select': self.parse_select(sql['select']),
            'where': self.parse_nested_cond(sql['where']), # 
            'group_by': [self.parse_col_unit(u) for u in sql['groupBy']],
            'order_by': self.parse_order_by(sql['orderBy']),
            'having': self.parse_cond(sql['having']),
            'limit': sql['limit'] if self.include_literals else (sql['limit'] is not None),
            'intersect': self.parse_sql(sql['intersect']),
            'except': self.parse_sql(sql['except']),
            'union': self.parse_sql(sql['union']),
            **({
                    'from': self.parse_from(sql['from'], self.infer_from_conditions),
                } if self.output_from else {})
        })

    def parse_select(self, select):
        is_distinct, aggs = select
        return {
            '_type': 'select',
            'is_distinct': is_distinct,
            'aggs': [self.parse_agg(agg) for agg in aggs],
        }

    def parse_agg(self, agg):
        agg_id, val_unit = agg
        return {
            '_type': 'agg',
            'agg_id': {'_type': self.AGG_TYPES_F[agg_id]},
            'val_unit': self.parse_val_unit(val_unit),
        }

    def parse_from(self, from_, infer_from_conditions=False):
        return filter_nones({
            '_type': 'from',
            'table_units': [
                self.parse_table_unit(u) for u in from_['table_units']],
            'conds': self.parse_cond(from_['conds']) \
                if not infer_from_conditions else None,
        })

    def parse_order_by(self, order_by):
        if not order_by:
            return None

        order, val_units = order_by
        return {
            '_type': 'order_by',
            'order': {'_type': self.ORDERS_F[order]},
            'val_units': [self.parse_val_unit(v) for v in val_units]
        }

    def parse_sql_query(self, query):
        print('len query: ', len(query))
        print('check query: ', query, '\n')
        query_op, val1, val2 = query
        result = {
                '_type': self.QUERY_OPERATORS_F[query_op],
                'val1': self.parse_val(val1)}
        if val2:
            result['val2'] = self.parse_val(val2)
        return result

    COND_TYPES_F, COND_TYPES_B = bimap(
        # ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'notnull', 'isnull'),
        # (None, 'Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like', 'Is', 'Exists', 'Notnull', 'Isnull'))
        range(1, 14),
        ('Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like', 'Is', 'Exists', 'Notnull', 'Isnull'))

    UNIT_TYPES_F, UNIT_TYPES_B = bimap(
        # ('none', '-', '+', '*', '/', 'abs', 'sum'),
        range(7),
        ('Column', 'Minus', 'Plus', 'Times', 'Divide', 'AbsMinus', 'SumMinus'))

    AGG_TYPES_F, AGG_TYPES_B = bimap(
        range(8),
        ('NoneAggOp', 'Max', 'Min', 'Count', 'Sum', 'Avg', 'Julianday', 'Length'))

    ORDERS_F, ORDERS_B = bimap(
        ('asc', 'desc'),
        ('Asc', 'Desc'))

    LOGIC_OPERATORS_F, LOGIC_OPERATORS_B = bimap(
        ('and', 'or'),
        ('And', 'Or'))
    
    # new
    DUAL_VAL_OPERATORS_F, DUAL_VAL_OPERATORS_B = bimap(
        range(5),
        # ('none', '-', '+', "*", '/', '>', '<', '>=','<='))
        ('DValue', 'DMinus', 'DPlus', "DTimes", 'DDivide'))

    QUERY_OPERATORS_F, QUERY_OPERATORS_B = bimap(
        range(14),
        # ('none', '-', '+', '>', '<', '>=','<=', '!=', '=', 'notnull', 'isnull', 'abs', 'min', 'max'))
        ('Single', 'QMinus', 'QPlus', 'QGT', 'QST', 'QGE','QSE', 'QNE', 'QE', 'QNOTNULL', 'QISNULL', 'QABS', 'QMIN', 'QMAX'))

    LOGIC_OPERATORS_NEST_F, LOGIC_OPERATORS_NEST_B = bimap(
        ('and', 'or'),
        ('NAnd', 'NOr'))

@attr.s
class SquallUnparser:
    ast_wrapper = attr.ib()
    schema = attr.ib()
    factorize_sketch = attr.ib(default=0)

    UNIT_TYPES_B = {
        'Minus': '-',
        'Plus': '+',
        'Times': '*',
        'Divide': '/',
        'Abs': 'abs',
        'Sum': 'sum'
    }
    COND_TYPES_B = {
        'Between': 'BETWEEN',
        'Eq': '=',
        'Gt': '>',
        'Lt': '<',
        'Ge': '>=',
        'Le': '<=',
        'Ne': '!=',
        'In': 'IN',
        'Like': 'LIKE',
        'Is': 'IS',
        'Isnull': 'IS NULL',
        'Notnull': 'NOT NULL'
    }

    @classmethod
    def conjoin_conds(cls, conds):
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {'_type': 'And', 'left': conds[0], 'right': cls.conjoin_conds(conds[1:])}

    @classmethod
    def linearize_cond(cls, cond):
        if cond['_type'] in ('And', 'Or'):
            conds, keywords = cls.linearize_cond(cond['right'])
            return [cond['left']] + conds, [cond['_type']] + keywords
        else:
            return [cond], []

    def unparse_val(self, val):
        if val['_type'] == 'Terminal':
            return "'terminal'"
        if val['_type'] == 'String':
            return val['s']
        if val['_type'] == 'ColUnit':
            return self.unparse_col_unit(val['c'])
        if val['_type'] == 'Number':
            return str(val['f'])
        if val['_type'] == 'ValSql':
            return f'({self.unparse_sql(val["s"])})'
        if val['_type'] == 'Present_ref':
            return "'present_ref'"

    def unparse_col_unit(self, col_unit):
        if 'col_id' in col_unit:
            column = self.schema.columns[col_unit['col_id']]
            if column.table is None:
                column_name = column.orig_name
            else:
                column_name = f'{column.table.orig_name}.{column.orig_name}'
        else:
            column_name = 'some_col'

        if col_unit['is_distinct']:
            column_name = f'DISTINCT {column_name}'
        agg_type = col_unit['agg_id']['_type']
        if agg_type == 'NoneAggOp':
            return column_name
        else:
            return f'{agg_type}({column_name})'

    def unparse_val_unit(self, val_unit):
        if val_unit['_type'] == 'Column':
            return self.unparse_col_unit(val_unit['col_unit1'])
        col1 = self.unparse_col_unit(val_unit['col_unit1'])
        col2 = self.unparse_col_unit(val_unit['col_unit2'])
        return f'{col1} {self.UNIT_TYPES_B[val_unit["_type"]]} {col2}'

    # def unparse_table_unit(self, table_unit):
    #    raise NotImplementedError

    def unparse_cond(self, cond, negated=False):
        if cond['_type'] == 'And':
            assert not negated
            return f'{self.unparse_cond(cond["left"])} AND {self.unparse_cond(cond["right"])}'
        elif cond['_type'] == 'Or':
            assert not negated
            return f'{self.unparse_cond(cond["left"])} OR {self.unparse_cond(cond["right"])}'
        elif cond['_type'] == 'Not':
            return self.unparse_cond(cond['c'], negated=True)
        elif cond['_type'] == 'Between':
            tokens = [self.unparse_val_unit(cond['val_unit'])]
            if negated:
                tokens.append('NOT')
            tokens += [
                'BETWEEN',
                self.unparse_val(cond['val1']),
                'AND',
                self.unparse_val(cond['val2']),
            ]
            return ' '.join(tokens)
        tokens = [self.unparse_val_unit(cond['val_unit'])]
        if negated:
            tokens.append('NOT')
        tokens += [self.COND_TYPES_B[cond['_type']], self.unparse_val(cond['val1'])]
        return ' '.join(tokens)

    def refine_from(self, tree):
        """
        1) Inferring tables from columns predicted 
        2) Mix them with the predicted tables if any
        3) Inferring conditions based on tables 
        """

        # nested query in from clause, recursively use the refinement
        if "from" in tree and tree["from"]["table_units"][0]["_type"] == 'TableUnitSql':
            for table_unit in tree["from"]["table_units"]:
                subquery_tree = table_unit["s"]
                self.refine_from(subquery_tree)
            return

        # get predicted tables
        predicted_from_table_ids = set()
        if "from" in tree:
            table_unit_set = []
            for table_unit in tree["from"]["table_units"]:
                if table_unit["table_id"] not in predicted_from_table_ids:
                    predicted_from_table_ids.add(table_unit["table_id"])
                    table_unit_set.append(table_unit)
            tree["from"]["table_units"] = table_unit_set  # remove duplicate

        # Get all candidate columns
        candidate_column_ids = set(self.ast_wrapper.find_all_descendants_of_type(
            tree, 'column', lambda field: field.type != 'sql'))
        candidate_columns = [self.schema.columns[i] for i in candidate_column_ids]
        must_in_from_table_ids = set(
            column.table.id for column in candidate_columns if column.table is not None)

        # Table the union of inferred and predicted tables
        all_from_table_ids = must_in_from_table_ids.union(predicted_from_table_ids)
        if not all_from_table_ids:
            # TODO: better heuristic e.g., tables that have exact match
            all_from_table_ids = {0}

        covered_tables = set()
        candidate_table_ids = sorted(all_from_table_ids)
        start_table_id = candidate_table_ids[0]
        conds = []
        for table_id in candidate_table_ids[1:]:
            if table_id in covered_tables:
                continue
            try:
                path = nx.shortest_path(
                    self.schema.foreign_key_graph, source=start_table_id, target=table_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                covered_tables.add(table_id)
                continue

            for source_table_id, target_table_id in zip(path, path[1:]):
                if target_table_id in covered_tables:
                    continue
                all_from_table_ids.add(target_table_id)
                col1, col2 = self.schema.foreign_key_graph[source_table_id][target_table_id]['columns']
                conds.append({
                    '_type': 'Eq',
                    'val_unit': {
                        '_type': 'Column',
                        'col_unit1': {
                            '_type': 'col_unit',
                            'agg_id': {'_type': 'NoneAggOp'},
                            'col_id': col1,
                            'is_distinct': False
                        },
                    },
                    'val1': {
                        '_type': 'ColUnit',
                        'c': {
                            '_type': 'col_unit',
                            'agg_id': {'_type': 'NoneAggOp'},
                            'col_id': col2,
                            'is_distinct': False
                        }
                    }
                })
        table_units = [{'_type': 'Table', 'table_id': i} for i in sorted(all_from_table_ids)]

        tree['from'] = {
            '_type': 'from',
            'table_units': table_units,
        }
        cond_node = self.conjoin_conds(conds)
        if cond_node is not None:
            tree['from']['conds'] = cond_node

    def unparse_sql(self, tree):
        self.refine_from(tree)

        result = [
            # select select,
            self.unparse_select(tree['select']),
            # from from,
            self.unparse_from(tree['from']),
        ]

        def find_subtree(_tree, name):
            if self.factorize_sketch == 0:
                return _tree, _tree
            elif name in _tree:
                if self.factorize_sketch == 1:
                    return _tree[name], _tree[name]
                elif self.factorize_sketch == 2:
                    return _tree, _tree[name]
                else:
                    raise NotImplementedError

        tree, target_tree = find_subtree(tree, "sql_where")
        # cond? where,
        if 'where' in target_tree:
            result += [
                'WHERE',
                self.unparse_cond(target_tree['where'])
            ]

        tree, target_tree = find_subtree(tree, "sql_groupby")
        # col_unit* group_by,
        if 'group_by' in target_tree:
            result += [
                'GROUP BY',
                ', '.join(self.unparse_col_unit(c) for c in target_tree['group_by'])
            ]

        tree, target_tree = find_subtree(tree, "sql_orderby")
        # order_by? order_by,
        if 'order_by' in target_tree:
            result.append(self.unparse_order_by(target_tree['order_by']))

        tree, target_tree = find_subtree(tree, "sql_groupby")
        # cond? having,
        if 'having' in target_tree:
            result += ['HAVING', self.unparse_cond(target_tree['having'])]

        tree, target_tree = find_subtree(tree, "sql_orderby")
        # int? limit, 
        if 'limit' in target_tree:
            if isinstance(target_tree['limit'], bool):
                if target_tree['limit']:
                    result += ['LIMIT', '1']
            else:
                result += ['LIMIT', str(target_tree['limit'])]

        tree, target_tree = find_subtree(tree, "sql_ieu")
        # sql? intersect,
        if 'intersect' in target_tree:
            result += ['INTERSECT', self.unparse_sql(target_tree['intersect'])]
        # sql? except,
        if 'except' in target_tree:
            result += ['EXCEPT', self.unparse_sql(target_tree['except'])]
        # sql? union
        if 'union' in target_tree:
            result += ['UNION', self.unparse_sql(target_tree['union'])]

        return ' '.join(result)

    def unparse_select(self, select):
        tokens = ['SELECT']
        if select['is_distinct']:
            tokens.append('DISTINCT')
        tokens.append(', '.join(self.unparse_agg(agg) for agg in select.get('aggs', [])))
        return ' '.join(tokens)

    def unparse_agg(self, agg):
        unparsed_val_unit = self.unparse_val_unit(agg['val_unit'])
        agg_type = agg['agg_id']['_type']
        if agg_type == 'NoneAggOp':
            return unparsed_val_unit
        else:
            return f'{agg_type}({unparsed_val_unit})'

    def unparse_from(self, from_):
        if 'conds' in from_:
            all_conds, keywords = self.linearize_cond(from_['conds'])
        else:
            all_conds, keywords = [], []
        assert all(keyword == 'And' for keyword in keywords)

        cond_indices_by_table = collections.defaultdict(set)
        tables_involved_by_cond_idx = collections.defaultdict(set)
        for i, cond in enumerate(all_conds):
            for column in self.ast_wrapper.find_all_descendants_of_type(cond, 'column'):
                table = self.schema.columns[column].table
                if table is None:
                    continue
                cond_indices_by_table[table.id].add(i)
                tables_involved_by_cond_idx[i].add(table.id)

        output_table_ids = set()
        output_cond_indices = set()
        tokens = ['FROM']
        for i, table_unit in enumerate(from_.get('table_units', [])):
            if i > 0:
                tokens += ['JOIN']

            if table_unit['_type'] == 'TableUnitSql':
                tokens.append(f'({self.unparse_sql(table_unit["s"])})')
            elif table_unit['_type'] == 'Table':
                table_id = table_unit['table_id']
                tokens += [self.schema.tables[table_id].orig_name]
                output_table_ids.add(table_id)

                # Output "ON <cond>" if all tables involved in the condition have been output
                conds_to_output = []
                for cond_idx in sorted(cond_indices_by_table[table_id]):
                    if cond_idx in output_cond_indices:
                        continue
                    if tables_involved_by_cond_idx[cond_idx] <= output_table_ids:
                        conds_to_output.append(all_conds[cond_idx])
                        output_cond_indices.add(cond_idx)
                if conds_to_output:
                    tokens += ['ON']
                    tokens += list(intersperse(
                        'AND',
                        (self.unparse_cond(cond) for cond in conds_to_output)))
        return ' '.join(tokens)

    def unparse_order_by(self, order_by):
        return f'ORDER BY {", ".join(self.unparse_val_unit(v) for v in order_by["val_units"])} {order_by["order"]["_type"]}'
