from groq import Groq
import os
import re
from dotenv import load_dotenv

from .prompt_templates import ANSWER_SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set.")

client = Groq(api_key=api_key)


def clean_answer(text: str) -> str:
    """Clean accidental formatting without changing the meaning."""
    if not text:
        return ""

    text = text.strip()

    # Remove Markdown formatting.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^(Answer|Key finding|Main finding|Interpretation|Conclusion)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_conversation_context(conversation=None) -> str:
    """Pass recent turns to the answer model for follow-up continuity."""
    if not conversation:
        return "No previous conversation."

    parts = []

    for item in conversation[-6:]:
        parts.append(
            f"User question: {str(item.get('question', '')).strip()}\n"
            f"Previous answer: {str(item.get('answer', '')).strip()}\n"
            f"Previous evidence: {item.get('result', [])[:8] if isinstance(item.get('result', []), list) else item.get('result', [])}"
        )

    return "\n\n".join(parts)


_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*%?")


def _numbers_in(text: str):
    """Extract numeric tokens from text as floats (percent sign stripped)."""
    values = []
    for tok in _NUMBER_PATTERN.findall(text):
        cleaned = tok.rstrip("%").replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _grounded_values(result_df) -> set:
    """Every numeric value actually present in the SQL result, rounded for tolerant matching."""
    values = set()
    for col in result_df.select_dtypes(include="number").columns:
        for v in result_df[col].dropna():
            values.add(round(float(v), 2))
    return values


def is_grounded(answer: str, result_df, tolerance: float = 0.05) -> bool:
    """
    True if every number in the answer can be matched (within a small rounding
    tolerance) to a value that actually appears in the SQL result. This is
    what catches a model inventing or approximating a figure it wasn't given.
    """
    grounded = _grounded_values(result_df)
    if not grounded:
        return True

    for value in _numbers_in(answer):
        if not any(abs(value - g) <= tolerance for g in grounded):
            return False
    return True


def _deterministic_answer(question: str, result_df) -> str:
    """
    Zero-hallucination fallback: a plain sentence built directly from the
    dataframe, no LLM involved. Used only if the model can't produce a
    grounded explanation after one retry.
    """
    rows = []
    for _, row in result_df.head(5).iterrows():
        rows.append(
            "; ".join(f"{col.replace('_', ' ')}: {row[col]}" for col in result_df.columns)
        )
    facts = " | ".join(rows)
    return f"Based on the data returned for '{question}' — {facts}."


def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
        reasoning_effort="low",
    )
    return clean_answer(response.choices[0].message.content)


def generate_business_answer(
    question: str,
    sql: str,
    result_df,
    conversation=None,
) -> str:
    """
    Generate a business explanation from the live query result, verified
    against that result before being returned. No result values or
    question-specific interpretations are hardcoded.
    """
    if result_df is None or result_df.empty:
        return (
            "I couldn't find matching records for that question in the "
            "available credit-risk dataset."
        )

    result_text = result_df.to_string(index=False)
    context = build_conversation_context(conversation)

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        result_text=result_text,
    )

    answer = _call_llm(prompt)

    if answer and not is_grounded(answer, result_df):
        # One retry, explicit about the exact permitted numbers.
        allowed = ", ".join(str(v) for v in sorted(_grounded_values(result_df)))
        retry_prompt = (
            prompt
            + f"\n\nYour previous attempt used a number not present in the "
            f"data. The ONLY numbers you may use are: {allowed}. "
            f"Rewrite the explanation using only those values."
        )
        answer = _call_llm(retry_prompt)

        if not answer or not is_grounded(answer, result_df):
            return _deterministic_answer(question, result_df)

    if not answer:
        return (
            "I found matching data, but I could not generate a reliable "
            "business explanation for it."
        )

    return answer