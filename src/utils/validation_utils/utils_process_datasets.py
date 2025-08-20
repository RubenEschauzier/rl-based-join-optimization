import csv
from rdflib import Graph, URIRef, Literal


def read_csv_to_nt(location_csv, write_location_nt):
    # initialize graph
    g = Graph()

    with open(location_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue  # skip malformed rows
            s, p, o = row

            subj = URIRef(s.strip())
            pred = URIRef(p.strip())
            obj = URIRef(o.strip())

            g.add((subj, pred, obj))

    # serialize to N-Triples
    g.serialize(destination=write_location_nt, format="nt")
    print(f"Saved RDF graph to {write_location_nt}")


def read_tab_delim_txt_to_nt(location_txt, write_location_nt):

    # Base namespaces
    ENTITY_NS = "http://example.org/entity/"
    PRED_NS = "http://example.org/predicate/"

    g = Graph()

    with open(location_txt, newline="", encoding="utf-8") as f:
        # Detect separator (comma for csv, tab for txt)
        delimiter = "\t"

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            print(row)
            if len(row) != 3:
                continue

            s, p, o = row
            subj = URIRef(ENTITY_NS + s.strip())
            pred = URIRef(PRED_NS + p.strip())
            obj = URIRef(ENTITY_NS + o.strip())

            g.add((subj, pred, obj))

    g.serialize(destination=write_location_nt, format="nt")
    print(f"Saved RDF graph to {write_location_nt}")

def nt_to_data_graph():
    pass


if __name__ == "__main__":

    location_lubm = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\lubm.csv"
    output_lubm = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\lubm.nt"
    location_swdf = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\swdf.csv"
    output_swdf = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\swdf.nt"
    location_wikidata = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\wikidata5m_all_triplet.txt"
    output_wikidata = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\wikidata5m.nt"
    location_yago = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\yago.txt"
    output_yago = r"C:\Users\ruben\projects\benchmarks-rl-based-opt\data\yago.nt"

    # read_csv_to_nt(location_lubm, output_lubm)
    # read_csv_to_nt(location_swdf, output_swdf)
    # read_tab_delim_txt_to_nt(location_wikidata, output_wikidata)
    read_tab_delim_txt_to_nt(location_yago, output_yago)
