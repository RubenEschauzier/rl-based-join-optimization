from rdflib.plugins.sparql.processor import prepareQuery
import rdflib

"""
Query class based on rdflib query object
"""


class Query:
    def __init__(self, query: str):
        self.query_string: str = query
        self.prepared_query = prepareQuery(query)
        self.string_tp, self.rdflib_tp = self.deconstruct_to_triple_pattern()
        # Featurized queries will be constructed by the query featurizer class
        self.features = None
        self.query_graph_representation = None
        pass

    def deconstruct_to_triple_pattern(self):
        rdflib_triple_patterns: list[rdflib.term] = self.prepared_query.algebra.get('p').get('p').get('triples')
        string_triple_patterns: list[str] = []
        for triple_pattern in rdflib_triple_patterns:
            string_triple_pattern: str = ""
            for term in triple_pattern:
                if type(term) == rdflib.term.Variable:
                    string_triple_pattern += "{} ".format(term.n3())
                    pass
                elif type(term) == rdflib.term.URIRef:
                    string_triple_pattern += "{} ".format(term.n3())
                    pass
                elif type(term) == rdflib.term.Literal:
                    string_triple_pattern += "{} ".format(term.n3())
                    pass
                else:
                    print("Term not supported: {}".format(type(term)))
            string_triple_patterns.append(string_triple_pattern.strip() + " .")
        return string_triple_patterns, rdflib_triple_patterns
        pass

    def set_featurized(self, query_features):
        self.features = query_features
