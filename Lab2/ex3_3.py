"""Retrieval SciFact con encoder condiviso, training contrastivo e ranking."""
import math
import random
from collections import defaultdict

import torch
from torch.nn import functional as F
from tqdm import tqdm

from common.plots import curves
from common.runtime import load_checkpoint, parser, positive_int, setup, write_csv, write_json
from lab2.models import Retriever


def relevance(rows):
    result = defaultdict(set)
    for item in rows:
        if item["score"] > 0:
            result[str(item["query-id"])].add(str(item["corpus-id"]))
    return dict(result)


def tokenize(tokenizer, texts, max_length, device):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items() if key in ("input_ids", "attention_mask")}


@torch.no_grad()
def embed(model, tokenizer, texts, args, device):
    model.eval()
    result = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding", leave=False):
        result.append(model(**tokenize(tokenizer, texts[start:start + args.batch_size], args.max_length, device)).cpu())
    return torch.cat(result)


def evaluate(model, tokenizer, queries, corpus, qrels, args, device):
    document_ids, query_ids = list(corpus), list(qrels)
    docs = embed(model, tokenizer, list(corpus.values()), args, device)
    query_vectors = embed(model, tokenizer, [queries[q] for q in query_ids], args, device)
    rows, examples = [], []
    # Similarità sul corpus completo, a blocchi per limitare la memoria.
    for start in range(0, len(query_ids), args.batch_size):
        scores = query_vectors[start:start + args.batch_size] @ docs.T
        for offset, values in enumerate(scores):
            qid = query_ids[start + offset]
            ranked = [document_ids[i] for i in values.argsort(descending=True).tolist()]
            positions = [index + 1 for index, did in enumerate(ranked) if did in qrels[qid]]
            ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(qrels[qid]))))
            dcg = sum(1 / math.log2(rank + 1) for rank in positions if rank <= 10)
            rows.append(dict(query_id=qid, recall_at_1=sum(rank <= 1 for rank in positions) / len(qrels[qid]),
                             recall_at_5=sum(rank <= 5 for rank in positions) / len(qrels[qid]),
                             recall_at_10=sum(rank <= 10 for rank in positions) / len(qrels[qid]),
                             mrr=1 / min(positions) if positions else 0.0, ndcg_at_10=dcg / ideal))
            if len(examples) < 10:
                examples.append(dict(query_id=qid, query=queries[qid], relevant=sorted(qrels[qid]),
                                     top_5=[dict(document_id=did, text=corpus[did]) for did in ranked[:5]]))
    if not rows:
        raise ValueError("Nessuna query valutabile")
    metrics = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != "query_id"}
    return metrics, rows, examples


def main():
    p = parser(__doc__, epochs=5, batch_size=16)
    p.set_defaults(lr=2e-5)
    p.add_argument("--model", default="distilbert/distilbert-base-uncased")
    p.add_argument("--max-length", type=positive_int, default=256)
    p.add_argument("--temperature", type=float, default=0.05)
    args = p.parse_args()
    if args.temperature <= 0 or args.max_length > 512:
        p.error("Temperatura positiva e max-length non superiore a 512 richiesti")
    device, out = setup(args, "lab2", "ex3_3")
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    saved = load_checkpoint(args.checkpoint, device, "retriever") if args.checkpoint else None
    config = saved["config"] if saved else dict(model=args.model, max_length=args.max_length, temperature=args.temperature)
    args.max_length, args.temperature = config["max_length"], config["temperature"]
    corpus_data = load_dataset("BeIR/scifact", "corpus", split="corpus", cache_dir=str(args.data_dir / "datasets"))
    query_data = load_dataset("BeIR/scifact", "queries", split="queries", cache_dir=str(args.data_dir / "datasets"))
    judgments = load_dataset("BeIR/scifact-qrels", cache_dir=str(args.data_dir / "datasets"))
    corpus = {str(item["_id"]): (item["title"] + " " + item["text"]).strip() for item in corpus_data}
    queries = {str(item["_id"]): item["text"] for item in query_data}
    training_qrels, test_qrels = relevance(judgments["train"]), relevance(judgments["test"])
    ids = sorted(training_qrels)
    random.Random(args.seed).shuffle(ids)
    cut = max(1, int(0.9 * len(ids)))
    train_ids, val_ids = ids[:cut], ids[cut:]
    if not val_ids or set(training_qrels) & set(test_qrels):
        raise ValueError("Split query non valido o sovrapposizione fra training e test")
    write_json(out / "splits.json", dict(train=train_ids, validation=val_ids, test=sorted(test_qrels)))
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = Retriever(AutoModel.from_pretrained(config["model"])).to(device)
    if saved:
        model.load_state_dict(saved["model"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rows, best = [], -1.0
        for number in range(1, args.epochs + 1):
            model.train()
            random.shuffle(train_ids)
            losses = []
            for start in tqdm(range(0, len(train_ids), args.batch_size), desc=f"Epoca {number}"):
                qids = train_ids[start:start + args.batch_size]
                # Ogni query ha un positivo e un negativo esplicito; gestiamo più positivi
                # e documenti condivisi senza trasformarli in falsi negativi nel batch.
                dids = []
                for qid in qids:
                    dids.append(random.choice(sorted(training_qrels[qid])))
                    negative = random.choice(list(corpus))
                    while negative in training_qrels[qid]:
                        negative = random.choice(list(corpus))
                    dids.append(negative)
                q = model(**tokenize(tokenizer, [queries[qid] for qid in qids], args.max_length, device))
                d = model(**tokenize(tokenizer, [corpus[did] for did in dids], args.max_length, device))
                logits = q @ d.T / args.temperature
                positive = torch.tensor([[did in training_qrels[qid] for did in dids] for qid in qids], device=device)
                loss = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            metrics, _, _ = evaluate(model, tokenizer, queries, corpus, {qid: training_qrels[qid] for qid in val_ids}, args, device)
            row = dict(epoch=number, train_loss=sum(losses) / len(losses), **{f"val_{key}": value for key, value in metrics.items()})
            rows.append(row)
            write_csv(out / "metrics.csv", rows)
            print(row)
            if metrics["ndcg_at_10"] > best:
                best = metrics["ndcg_at_10"]
                torch.save(dict(kind="retriever", config=config, model=model.state_dict()), out / "best.pt")
        curves(rows, out / "loss.png", ["train_loss"])
        curves(rows, out / "ranking.png", ["val_mrr", "val_ndcg_at_10", "val_recall_at_10"])
        model.load_state_dict(load_checkpoint(out / "best.pt", device, "retriever")["model"])
    metrics, rows, examples = evaluate(model, tokenizer, queries, corpus, test_qrels, args, device)
    write_json(out / "test_metrics.json", metrics)
    write_csv(out / "test_queries.csv", rows)
    write_json(out / "retrieval_examples.json", examples)
    tokenizer.save_pretrained(out / "tokenizer")


if __name__ == "__main__":
    main()
