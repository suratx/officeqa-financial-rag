import os
import re
import json
import anthropic

MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def answer_question(question, contexts, engineered=False):
    """
    contexts: list of dicts {text, filename, year, month}
    Returns the raw answer string.
    """
    if engineered:
        blocks = "\n\n".join(
            f"[Source: {c['filename']} | Year: {c['year']} | Month: {c['month']}]\n{c['text']}"
            for c in contexts
        )
        system = (
            "You are a financial analyst answering questions strictly from the "
            "provided U.S. Treasury Bulletin excerpts. Only use facts present in "
            "the context below. For every number you state, note which source "
            "excerpt it came from. If the context does not contain the answer, "
            "say so explicitly rather than guessing."
        )
    else:
        blocks = "\n\n".join(c["text"] for c in contexts)
        system = "Answer the question using the context provided."

    msg = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Context:\n{blocks}\n\nQuestion: {question}\n\n"
                       f"Give a direct final answer."
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def extract_claims(answer_text):
    """Split answer into sentence-level claims worth checking (those containing
    a number or a capitalized entity), used for groundedness scoring."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    claims = [s for s in sentences if re.search(r"\d", s) and len(s.strip()) > 0]
    return claims or sentences


def check_claim_supported(claim, contexts):
    context_text = "\n\n".join(c["text"] for c in contexts)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=10,
        system="Answer only 'yes' or 'no'.",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context_text}\n\nClaim: {claim}\n\n"
                       f"Is this claim directly supported by the context above?"
        }],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip().lower()
    return text.startswith("y")
