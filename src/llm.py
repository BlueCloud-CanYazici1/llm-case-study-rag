# ------------------------------------------------------------
# Local LLM Utilities
#
# This module uses a local LLM in a tightly constrained way.
# The model does not perform open-ended generation.
# Instead, it rewrites retrieved evidence into a short,
# grounded answer sentence.
# ------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_llm(model_name: str):
    """
    Load a local lightweight text-generation model once at startup.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return generator


def build_rewrite_prompt(question: str, evidence: str) -> str:
    """
    Build a tightly constrained prompt that asks the LLM
    to rewrite evidence into one short answer sentence.
    """

    return f"""Rewrite the evidence into one short, natural answer sentence.

Rules:
- Use only the provided evidence.
- Do not add any new facts.
- Do not use outside knowledge.
- Keep the answer short and directly responsive to the question.
- Do not start with phrases like "The answer is".
- If the evidence is insufficient, say:
  "The provided documents do not contain enough information to answer this question reliably."

Question:
{question}

Evidence:
{evidence}

Answer:
"""


def clean_generated_answer(answer: str) -> str:
    """
    Remove prompt artifacts and keep only the first clean line.
    """

    answer = answer.strip()

    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[-1].strip()

    answer = answer.split("\n")[0].strip()

    return answer


def rewrite_answer_with_llm(generator, question: str, evidence: str) -> str:
    """
    Rewrite the supporting evidence into a concise answer sentence.
    """

    if not evidence.strip():
        return "The provided documents do not contain enough information to answer this question reliably."

    prompt = build_rewrite_prompt(question, evidence)

    output = generator(
        prompt,
        max_new_tokens=30,
        do_sample=False,
        return_full_text=False,
    )

    answer = output[0]["generated_text"]
    return clean_generated_answer(answer)