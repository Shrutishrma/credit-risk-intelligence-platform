from groq import Groq
import os
import re
from dotenv import load_dotenv

from .prompt_templates import ANSWER_SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE

load_dotenv()


def _get_secret(name: str):
    """Read from environment variables or Streamlit Secrets."""
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
    """Clean accidental formatting without changing meaning."""
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)

    text = re.sub(
        r"^(Answer|Key finding|Main finding|Interpretation|Conclusion)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

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
    """Extract numeric values from generated text."""
    values = []

    for token in _NUMBER_PATTERN.findall(text):
        cleaned = token.rstrip("%").replace(",", "")

        try:
            values.append(float(cleaned))
        except ValueError:
            continue

    return values


def _grounded_values(result_df) -> set:
    """Return numeric values actually present in the SQL result."""

    values = set()

    for column in result_df.select_dtypes(include="number").columns:
        for value in result_df[column].dropna():
            values.add(round(float(value), 2))

    return values


def is_grounded(answer: str, result_df, tolerance: float = 0.05) -> bool:
    """
    Check that every numerical value mentioned by the LLM exists
    in the returned result.
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
    Generate the final business explanation.

    GPT-OSS reasoning is excluded so message.content contains the
    actual business answer rather than reasoning output.
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
        temperature=0.4,
        max_completion_tokens=700,
        reasoning_effort="low",
        include_reasoning=False,
        stream=False,
    )

    content = getattr(response.choices[0].message, "content", None)

    return clean_answer(content or "")


def generate_business_answer(
    question: str,
    sql: str,
    result_df,
    conversation=None,
) -> str:
    """
    Generate a natural business explanation from the live query result.

    The LLM receives only the actual returned evidence and is instructed
    not to invent numerical values.
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

    prompt += """

Response requirements:
- Answer the user's question directly.
- Explain the result in natural business language.
- Use the actual group names and numerical values returned above.
- Do not invent numbers.
- Do not perform additional calculations or create new percentages.
- Do not mention SQL, tables, databases, prompts, models, or implementation.
- Do not simply repeat the raw rows.
- Write 3 to 4 natural sentences.
- State the main finding first.
- Then explain what the finding means.
- End with a cautious business takeaway.
- When comparing groups, clearly name which group is higher or lower.
- When the question asks "why", describe an observed association and do not claim causality.
"""

    answer = _call_llm(prompt)

    # Retry once if the answer contains unsupported numerical values.
    if answer and not is_grounded(answer, result_df):

        retry_prompt = prompt + """

The previous answer could not be verified because it introduced a
numerical value that was not present in the returned evidence.

Rewrite the answer using only numerical values that appear exactly
in the returned evidence above. Do not calculate differences or
introduce any new numbers.
"""

        answer = _call_llm(retry_prompt)

    if not answer:
        return (
            "I found the requested result, but the AI explanation could "
            "not be generated right now. Please try the question again."
        )

    if not is_grounded(answer, result_df):
        return (
            "I found the requested result, but I could not generate a "
            "numerically reliable business explanation. Please try again."
        )

    return answer