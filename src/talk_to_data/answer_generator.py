from groq import Groq
import os
import re
from dotenv import load_dotenv

from .prompt_templates import ANSWER_SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE

load_dotenv()


def _get_secret(name: str):
    """Read a setting from environment variables or Streamlit Secrets."""
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


api_key = _get_secret("GROQ_API_KEY")

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
            f"Previous evidence: "
            f"{item.get('result', [])[:8] if isinstance(item.get('result', []), list) else item.get('result', [])}"
        )

    return "\n\n".join(parts)


_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*%?")


def _numbers_in(text: str):
    """Extract numeric tokens from text as floats."""
    values = []

    for tok in _NUMBER_PATTERN.findall(text):
        cleaned = tok.rstrip("%").replace(",", "")

        try:
            values.append(float(cleaned))
        except ValueError:
            continue

    return values


def _grounded_values(result_df) -> set:
    """Return numeric values actually present in the SQL result."""
    values = set()

    for col in result_df.select_dtypes(include="number").columns:
        for value in result_df[col].dropna():
            values.add(round(float(value), 2))

    return values


def is_grounded(answer: str, result_df, tolerance: float = 0.05) -> bool:
    """
    Ensure every numerical value mentioned in the answer exists in the
    returned SQL result. This prevents invented figures.
    """

    grounded = _grounded_values(result_df)

    if not grounded:
        return True

    for value in _numbers_in(answer):
        if not any(abs(value - actual) <= tolerance for actual in grounded):
            return False

    return True


def _call_llm(prompt: str) -> str:
    """
    Generate a business explanation from Groq GPT-OSS.

    Reasoning is deliberately excluded from the returned message so the
    response content contains the actual business answer.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
        max_completion_tokens=800,
        reasoning_effort="low",
        include_reasoning=False,
        stream=False,
    )

    message = response.choices[0].message

    return clean_answer(
        getattr(message, "content", None) or ""
    )


def generate_business_answer(
    question: str,
    sql: str,
    result_df,
    conversation=None,
) -> str:
    """
    Generate a natural business explanation from the live SQL result.

    No answer values are hardcoded. The response is generated from the
    actual query result and checked for unsupported numerical claims.
    """

    if result_df is None or result_df.empty:
        return (
            "I couldn't find matching records for that question in the "
            "available credit-risk data."
        )

    result_text = result_df.to_string(index=False)

    context = build_conversation_context(conversation)

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        result_text=result_text,
    )

    # Make the numerical grounding requirement explicit.
    prompt += """

Additional response rules:
- Use the returned values exactly as provided.
- Do not invent numerical values.
- Do not calculate new numerical values or percentage differences.
- You may compare values using words such as higher, lower, largest, or smallest.
- Explain what the result means in practical business language.
- Do not mention SQL, database, query, dataframe, or implementation details.
- Return one natural paragraph of 3-4 sentences.
"""

    # First attempt.
    answer = _call_llm(prompt)

    # Retry if the model returned nothing or introduced unsupported numbers.
    if not answer or not is_grounded(answer, result_df):

        retry_prompt = prompt + """

The previous response could not be safely verified.

Rewrite the answer using ONLY the exact numerical values present in the
returned result. Do not introduce any new numerical calculations.
Return only the final business explanation.
"""

        answer = _call_llm(retry_prompt)

    # If Groq still fails, show a clear error rather than exposing a raw,
    # database-style fallback as if it were a finished business answer.
    if not answer:
        return (
            "I found the requested result, but I couldn't generate the "
            "business explanation right now. Please try the question again."
        )

    if not is_grounded(answer, result_df):
        return (
            "I found the requested result, but I couldn't generate a "
            "numerically reliable explanation for it. Please try again."
        )

    return answer