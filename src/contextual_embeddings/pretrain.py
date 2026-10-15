"""Pretrain the contextualizer on the counterfactual corpus.

    python -m src.contextual_embeddings.pretrain [hydra overrides]

Builds the vocabulary (once), splits the corpus by HELD-OUT CONSTANTS (instances with a constant
from a random `heldout_constant_share` of entities only validate, so validation measures what
transfers to unseen constants), trains with
    BCE(which arm was replaced) + BCE(instance is empty) + table_l2 * ||looked-up vectors - rdf2vec||^2
and keeps the epoch with the best `training.selection_metric` on validation (default: AUROC of
"empty" on positives vs near-miss negatives). Writes training.output_directory/contextualizer.pt (state, hyperparameters,
vocabulary) and metrics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from src.contextual_embeddings import vocabulary as vocab
from src.contextual_embeddings.model import Contextualizer, collate
from src.contextual_embeddings.profiles import ProfileTable


def _token_maps(terms, predicates):
    index = {t: i for i, t in enumerate(terms)}
    prefix = len(vocab.entity_term(0)) - 1
    entity_token = {int(t[prefix:]): i for t, i in index.items() if t != vocab.UNKNOWN and t[prefix:].isdigit()}
    predicate_token = np.array([index.get(p, 0) for p in predicates], dtype=np.int64)
    return entity_token, predicate_token


def _split(instances, share, seed):
    def held(e):
        return (e * 2654435761 + seed) % 1_000_003 < share * 1_000_003
    val = [i for i, inst in enumerate(instances) if any(e >= 0 and held(e) for e in inst["nodes"])]
    val_set = set(val)
    return [i for i in range(len(instances)) if i not in val_set], val


@torch.no_grad()
def evaluate(model, instances, entity_token, predicate_token, device, batch_size=1024, profiles=None):
    model.eval()
    empty_scores, kinds, arm_hits, arm_total = [], [], 0, 0
    for start in range(0, len(instances), batch_size):
        chunk = instances[start:start + batch_size]
        batch = collate(chunk, entity_token, predicate_token, device, profiles)
        _, arm_logits, empty_logits = model(batch)
        empty_scores += empty_logits.sigmoid().cpu().tolist()
        kinds += [inst["kind"] for inst in chunk]
        position = 0
        for inst in chunk:
            n = len(inst["edges"])
            if inst["empty"] and any(inst["replaced"]):
                arm_total += 1
                arm_hits += bool(inst["replaced"][int(arm_logits[position:position + n].argmax())])
            position += n
    scores, kinds = np.array(empty_scores), np.array(kinds)
    positive = kinds == "positive"

    def auroc(negative):
        mask = positive | negative
        return float(roc_auc_score(~positive[mask], scores[mask])) if negative.any() and positive.any() else float("nan")

    return {"auroc_all": auroc(~positive),
            "auroc_near_miss": auroc(np.char.startswith(kinds.astype(str), "near_miss")),
            "auroc_near_miss_constant": auroc(kinds == "near_miss_constant"),
            "auroc_near_miss_predicate": auroc(kinds == "near_miss_predicate"),
            "auroc_easy": auroc(kinds == "easy"),
            "replaced_arm_top1": arm_hits / max(arm_total, 1)}


def pretrain(cfg):
    torch.manual_seed(int(cfg.training.seed))
    gen = OmegaConf.load(cfg.corpus.generation_config).generation
    predicates = vocab.predicate_terms(Path(gen.cache_directory) / "graph.npz")
    instances = vocab.load_corpus(cfg.corpus.output_directory)
    print(f"[pretrain] {len(instances):,} corpus instances", flush=True)
    vocab_dir = Path(cfg.vocabulary.directory)
    if (vocab_dir / "vocabulary.json").exists():
        terms, init = vocab.load(vocab_dir)
    else:
        terms, init = vocab.build(instances, predicates, cfg.vocabulary.queries, cfg.vocabulary.rdf2vec,
                                  vocab_dir, int(cfg.vocabulary.min_count))
    entity_token, predicate_token = _token_maps(terms, predicates)
    train_ids, val_ids = _split(instances, float(cfg.training.heldout_constant_share), int(cfg.training.seed))
    val = [instances[i] for i in val_ids]
    print(f"[pretrain] train {len(train_ids):,}, validation (held-out constants) {len(val):,}", flush=True)

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    profiles = ProfileTable(cfg.profiles.directory) if cfg.model.get("profiles", False) else None
    model = Contextualizer(init, int(cfg.model.hidden), int(cfg.model.layers), float(cfg.model.get("dropout", 0.0)),
                           profile_types=None if profiles is None else profiles.type_features,
                           profile_hidden=int(cfg.model.get("profile_hidden", 64)),
                           profile_attention=cfg.model.get("profile_attention", None)).to(device)
    freeze_table = bool(cfg.training.get("freeze_table", False))
    if freeze_table:                 # rdf2vec stays intact; nothing can be memorised per constant
        model.table.weight.requires_grad = False
    n_dense = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("table"))
    print(f"[pretrain] contextualizer network: {n_dense:,} parameters (+ token table {init.shape[0]:,} x {init.shape[1]})")
    sparse = None if freeze_table else torch.optim.SparseAdam(list(model.table.parameters()),
                                                              lr=float(cfg.training.table_lr))
    dense = torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith("table")],
                              lr=float(cfg.training.lr))
    bce = torch.nn.BCEWithLogitsLoss()
    out_dir = Path(cfg.training.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.training.seed))
    history, best, since_best = [], -1.0, 0
    patience = int(cfg.training.get("early_stopping_patience", 10 ** 9))
    selection = str(cfg.training.get("selection_metric", "auroc_near_miss"))
    for epoch in range(int(cfg.training.epochs)):
        model.train()
        order = rng.permutation(train_ids)
        losses, start = [], time.perf_counter()
        for b in range(0, len(order), int(cfg.training.batch_size)):
            batch = collate([instances[i] for i in order[b:b + int(cfg.training.batch_size)]],
                            entity_token, predicate_token, device, profiles)
            _, arm_logits, empty_logits = model(batch)
            loss = (float(cfg.training.arm_weight) * bce(arm_logits, batch["replaced"])
                    + float(cfg.training.empty_weight) * bce(empty_logits, batch["empty"]))
            if not freeze_table:
                loss = loss + float(cfg.training.table_l2) * model.table_l2(
                    torch.unique(torch.cat([batch["node_token"], batch["pattern_token"]])))
            dense.zero_grad()
            if sparse is not None:
                sparse.zero_grad()
            loss.backward()
            dense.step()
            if sparse is not None:
                sparse.step()
            losses.append(loss.item())
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.perf_counter() - start,
                   **evaluate(model, val, entity_token, predicate_token, device, profiles=profiles)}
        history.append(metrics)
        print(f"[pretrain] {json.dumps(metrics)}", flush=True)
        if metrics[selection] > best:
            best, since_best = metrics[selection], 0
            torch.save({"state_dict": {k: v for k, v in model.state_dict().items()},
                        "hidden": int(cfg.model.hidden), "layers": int(cfg.model.layers),
                        "dropout": float(cfg.model.get("dropout", 0.0)),
                        "profiles_directory": None if profiles is None else str(cfg.profiles.directory),
                        "profile_hidden": int(cfg.model.get("profile_hidden", 64)),
                        "profile_attention": cfg.model.get("profile_attention", None), "predicate_terms": predicates,
                        "terms": terms, "metrics": metrics}, out_dir / "contextualizer.pt")
        else:
            since_best += 1
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)
        if since_best >= patience:
            print(f"[pretrain] early stop: no improvement for {patience} epochs")
            break
    print(f"[pretrain] best validation {selection} {best:.3f} -> {out_dir / 'contextualizer.pt'}")


@hydra.main(version_base=None, config_path="../../experiments/experiment_configs/contextual_embeddings",
            config_name="contextualizer_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    pretrain(cfg)


if __name__ == "__main__":
    main()
