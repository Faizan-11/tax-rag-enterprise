"""
06_eval.py
----------
Evaluation harness for the Enterprise Tax RAG.

Measures:
- Retrieval accuracy (correct pages found?)
- Keyword match (expected terms present?)
- Faithfulness & relevancy (LLM-as-judge)
- Refusal correctness (tricks correctly refused?)
- Latency
"""

import sys
import csv
import json
import time
import os
import pickle
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flashrank import Ranker, RerankRequest

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, JUDGE_MODEL, TOP_K_RETRIEVAL, TOP_K_FINAL

# Init
client = QdrantClient(url=QDRANT_URL)
embed_model = SentenceTransformer(EMBEDDING_MODEL)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

INPUT_CSV = DATA_DIR / "eval" / "questions.csv"
RESULTS_JSONL = DATA_DIR / "eval" / "results.jsonl"
SUMMARY_JSON = DATA_DIR / "eval" / "summary.json"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the Indian Income Tax Act.
Answer ONLY using the provided context passages.
If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
Cite page numbers in the format [pages X-Y] after each fact.
Be concise."""

JUDGE_PROMPT = """You are a strict evaluator. Given a QUESTION, CONTEXT, and ANSWER, score two things.

FAITHFULNESS: Is every claim in the ANSWER supported by the CONTEXT?
RELEVANCY: Does the ANSWER address the QUESTION?

Respond EXACTLY in this format:
FAITHFULNESS: <0.0 to 1.0>
RELEVANCY: <0.0 to 1.0>
REASONING: <one short sentence>
"""


def load_bm25():
    with (DATA_DIR / "bm25_index.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    chunks = []
    with (DATA_DIR / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return bm25, chunks


def rewrite_query(question: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in Indian tax law. Rewrite the user's question into a formal search query. Do NOT assume the document year. Return ONLY the rewritten query."},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question  # fallback


def hybrid_search(query: str, bm25, bm25_chunks):
    q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = client.query_points(
        collection_name=COLLECTION_NAME, query=q_emb, limit=TOP_K_RETRIEVAL
    ).points

    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_RETRIEVAL]

    combined = {}
    for hit in vector_results:
        combined[hit.id] = {"chunk_id": hit.id, "text": hit.payload["text"], "pages": hit.payload["pages"], "v_score": hit.score, "b_score": 0.0}
    for idx in top_bm25_indices:
        chunk = bm25_chunks[idx]
        cid = chunk["chunk_id"]
        if cid not in combined:
            combined[cid] = {"chunk_id": cid, "text": chunk["text"], "pages": chunk["pages"], "v_score": 0.0, "b_score": bm25_scores[idx]}
        else:
            combined[cid]["b_score"] = bm25_scores[idx]

    return sorted(combined.values(), key=lambda x: (x["v_score"] > 0) + (x["b_score"] > 0) + x["v_score"], reverse=True)[:TOP_K_RETRIEVAL]


def rerank(question: str, hits: list):
    passages = [{"id": str(h["chunk_id"]), "text": h["text"], "meta": h} for h in hits]
    results = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [r["meta"] for r in results[:TOP_K_FINAL]]


def generate(question: str, hits: list) -> str:
    context = "\n\n".join([f"--- Passage {i} [pages {h['pages']}] ---\n{h['text']}" for i, h in enumerate(hits, 1)])
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def judge(question: str, context: str, answer: str) -> dict:
    if "don't have enough information" in answer.lower() or "error" in answer.lower():
        return {"faithfulness": None, "relevancy": None, "reasoning": "refusal_or_error"}
    try:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:3000]}\n\nANSWER:\n{answer[:1500]}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        import re
        faith = re.search(r"FAITHFULNESS:\s*([0-9.]+)", raw)
        rel = re.search(r"RELEVANCY:\s*([0-9.]+)", raw)
        reas = re.search(r"REASONING:\s*(.+)", raw)
        return {
            "faithfulness": float(faith.group(1)) if faith else None,
            "relevancy": float(rel.group(1)) if rel else None,
            "reasoning": reas.group(1).strip()[:200] if reas else "parse_error",
        }
    except Exception as e:
        return {"faithfulness": None, "relevancy": None, "reasoning": f"judge_error: {e}"}


def check_retrieval(hits, expected_pages):
    if not expected_pages:
        return None
    found = set()
    for h in hits:
        found.update(h.get("pages", []))
    return bool(found & set(expected_pages))


def check_keywords(answer, expected):
    if expected.strip().upper() == "REFUSE":
        return "don't have enough information" in answer.lower()
    kws = [k.strip().lower() for k in expected.split(";") if k.strip()]
    if not kws:
        return True
    matched = sum(1 for k in kws if k in answer.lower())
    return matched / len(kws) >= 0.6


def main():
    if not INPUT_CSV.exists():
        print(f"❌ Create {INPUT_CSV} first")
        return

    print(f"Loading BM25 index and chunks...")
    bm25, bm25_chunks = load_bm25()

    questions = []
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    print(f"Loaded {len(questions)} questions\n")
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with RESULTS_JSONL.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(questions, start=1):
            qid = row["id"]
            question = row["question"]
            expected_kw = row["expected_keywords"]
            expected_pages = [int(p.strip()) for p in row["expected_pages"].split(",") if p.strip()] if row["expected_pages"].strip() else []
            category = row["category"]

            print(f"[{i}/{len(questions)}] Q{qid} ({category}): {question[:60]}...")

            t0 = time.time()
            try:
                rewritten = rewrite_query(question)
                hits = hybrid_search(rewritten, bm25, bm25_chunks)
                final_hits = rerank(question, hits)
                answer = generate(question, final_hits)
                latency = time.time() - t0

                context = "\n\n".join([h["text"] for h in final_hits])
                judged = judge(question, context, answer)

                result = {
                    "id": qid,
                    "question": question,
                    "category": category,
                    "rewritten": rewritten,
                    "retrieval_ok": check_retrieval(final_hits, expected_pages),
                    "keyword_ok": check_keywords(answer, expected_kw),
                    "faithfulness": judged["faithfulness"],
                    "relevancy": judged["relevancy"],
                    "judge_reasoning": judged["reasoning"],
                    "latency_sec": round(latency, 2),
                    "answer": answer,
                }
            except Exception as e:
                result = {
                    "id": qid, "question": question, "category": category,
                    "rewritten": None, "retrieval_ok": None, "keyword_ok": False,
                    "faithfulness": None, "relevancy": None,
                    "judge_reasoning": f"pipeline_error: {e}",
                    "latency_sec": round(time.time() - t0, 2), "answer": "",
                }

            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            status = []
            if result["retrieval_ok"] is not None:
                status.append(f"ret={'✅' if result['retrieval_ok'] else '❌'}")
            status.append(f"kw={'✅' if result['keyword_ok'] else '❌'}")
            if result["faithfulness"] is not None:
                status.append(f"faith={result['faithfulness']:.2f}")
            status.append(f"{result['latency_sec']}s")
            print(f"    {' | '.join(status)}")

    # Aggregate
    total = len(results)
    judged = [r for r in results if r["faithfulness"] is not None]
    refusals = [r for r in results if (r["judge_reasoning"] or "") == "refusal_or_error" or "don't have enough" in (r["answer"] or "").lower()]

    summary = {
        "total": total,
        "judged_successfully": len(judged),
        "refusals": len(refusals),
        "judge_failures": total - len(judged) - len(refusals),
        "retrieval_accuracy": (
            sum(1 for r in results if r["retrieval_ok"]) / max(1, sum(1 for r in results if r["retrieval_ok"] is not None))
            if any(r["retrieval_ok"] is not None for r in results) else None
        ),
        "keyword_match_rate": sum(1 for r in results if r["keyword_ok"]) / total,
        "avg_faithfulness": sum(r["faithfulness"] for r in judged) / len(judged) if judged else None,
        "avg_relevancy": sum(r["relevancy"] for r in judged) / len(judged) if judged else None,
        "avg_latency": sum(r["latency_sec"] for r in results) / total,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:       {summary['total']}")
    print(f"Judged successfully:   {summary['judged_successfully']}")
    print(f"Refusals:              {summary['refusals']}")
    print(f"Judge failures:        {summary['judge_failures']}")
    if summary["retrieval_accuracy"] is not None:
        print(f"Retrieval accuracy:    {summary['retrieval_accuracy']:.1%}")
    print(f"Keyword match rate:    {summary['keyword_match_rate']:.1%}")
    if summary["avg_faithfulness"]:
        print(f"Avg faithfulness:      {summary['avg_faithfulness']:.3f}")
        print(f"Avg relevancy:         {summary['avg_relevancy']:.3f}")
    print(f"Avg latency:           {summary['avg_latency']:.2f}s")


if __name__ == "__main__":
    main()