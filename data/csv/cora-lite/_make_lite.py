import csv, json, os

SRC = r"D:\VMwareShareFolder\ccnu\Fusion\GNN\scripts\gcn-pytorch\data\csv\cora"
DST = r"/csv-cora-lite"
LIMIT = 250
os.makedirs(DST, exist_ok=True)

# nodes
nodes = []
with open(os.path.join(SRC, "nodes.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if int(r["node_id"]) < LIMIT:
            nodes.append(r)
with open(os.path.join(DST, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["node_id", "split", "label"])
    w.writeheader(); w.writerows(nodes)

# features (sparse)
feats = []
with open(os.path.join(SRC, "features.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if int(r["node_id"]) < LIMIT:
            feats.append(r)
with open(os.path.join(DST, "features.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["node_id", "feature_id", "value"])
    w.writeheader(); w.writerows(feats)

# features dense
dense = []
with open(os.path.join(SRC, "features_dense.csv"), newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    hdr = rd.fieldnames
    for r in rd:
        if int(r["node_id"]) < LIMIT:
            dense.append(r)
with open(os.path.join(DST, "features_dense.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader(); w.writerows(dense)

# edges: both ends in range
edges = []
with open(os.path.join(SRC, "edges.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if int(r["source"]) < LIMIT and int(r["target"]) < LIMIT:
            edges.append(r)
with open(os.path.join(DST, "edges.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source", "target", "weight"])
    w.writeheader(); w.writerows(edges)

with open(os.path.join(SRC, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)

splits = {}
for r in nodes:
    splits[r["split"]] = splits.get(r["split"], 0) + 1
meta["num_nodes"] = len(nodes)
meta["num_edges"] = len(edges)
meta["num_nonzero_features"] = len(feats)
meta["splits"] = splits
with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("nodes", len(nodes), "edges", len(edges), "feat_rows", len(feats), "splits", splits)