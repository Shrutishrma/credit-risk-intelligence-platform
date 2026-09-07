from groq import Groq
import os
import re
from dotenv import load_dotenv

from .prompt_templates import SQL_SYSTEM_PROMPT, SQL_SCHEMA_PROMPT

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


def clean_sql(sql: str) -> str:
    """Clean common formatting returned by the model."""
    if not sql:
        return ""

    sql = sql.strip()

    sql = re.sub(
        r"```sql\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.replace("```", "")

    sql = re.sub(
        r"^\s*SQL\s*:\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    return sql.strip()


def build_conversation_context(conversation=None) -> str:
    """
    Keep recent turns as context for follow-up questions.

    Actual previous results are passed in from the running application.
    No answer values are hardcoded here.
    """
    if not conversation:
        return "No previous conversation."

    sections = []

    for number, item in enumerate(conversation[-6:], start=1):

        question = str(
            item.get("question", "")
        ).strip()

        answer = str(
            item.get("answer", "")
        ).strip()

        sql = str(
            item.get("sql", "")
        ).strip()

        result = item.get(
            "result",
            [],
        )

        if isinstance(result, list):
            result = result[:8]

        sections.append(
            f"""
Turn {number}
User question:
{question}

Previous answer:
{answer}

Previous SQL:
{sql}

Previous evidence:
{result}
""".strip()
        )

    return "\n\n".join(sections)


def likely_follow_up(question: str, conversation=None) -> bool:
    """
    Identify whether context is likely useful.

    This does not decide what the question means; it only decides whether
    recent conversation should be included.
    """
    if not conversation:
        return False

    q = question.lower().strip()

    references = [
        "what about",
        "how about",
        "what does that",
        "what is that",
        "what about them",
        "those applicants",
        "those people",
        "same group",
        "same applicants",
        "compare that",
        "compare it",
        "is that higher",
        "is that lower",
        "are they",
        "why is that",
        "why are they",
        "how does that",
        "that rate",
        "that amount",
        "that result",
        "them",
        "they",
    ]

    return any(token in q for token in references)


def deterministic_sql(question: str):
    """
    Reliable SQL for a small set of very common, unambiguous questions.

    These are query templates, not hardcoded results. All values still come
    from the live applications table.
    """
    q = question.lower().strip()

    if (
        "overall default rate" in q
        or "default rate overall" in q
        or "overall rate of default" in q
    ):
        return """
SELECT
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM applications
"""

    if (
        "how many applicants defaulted" in q
        or "how many defaulted" in q
        or "number of defaults" in q
        or "count of defaults" in q
    ):
        return """
SELECT
    COUNT(*) AS defaulted_applications
FROM applications
WHERE TARGET = 1
"""

    if (
        "average credit amount" in q
        or "average loan amount" in q
        or "average credit" in q
    ) and "defaulted" not in q and "male" not in q and "female" not in q:
        return """
SELECT
    ROUND(AVG(AMT_CREDIT), 2) AS average_credit
FROM applications
"""

    if (
        "default rate by education" in q
        or ("education" in q and "default rate" in q)
    ):
        return """
SELECT
    NAME_EDUCATION_TYPE AS education,
    COUNT(*) AS applications,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM applications
GROUP BY NAME_EDUCATION_TYPE
HAVING COUNT(*) >= 1000
ORDER BY default_rate DESC
"""

    if (
        "default rate by income type" in q
        or ("income type" in q and "default rate" in q)
    ):
        return """
SELECT
    NAME_INCOME_TYPE AS income_type,
    COUNT(*) AS applications,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM applications
GROUP BY NAME_INCOME_TYPE
HAVING COUNT(*) >= 1000
ORDER BY default_rate DESC
"""

    if (
        "male" in q
        and "female" in q
        and "default" in q
    ):
        return """
SELECT
    CODE_GENDER AS gender,
    COUNT(*) AS applications,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM applications
WHERE CODE_GENDER IN ('M', 'F')
GROUP BY CODE_GENDER
HAVING COUNT(*) >= 1000
ORDER BY default_rate DESC
"""

    if (
        "previous application refusal rate" in q
        or "average refusal rate" in q
        or "average previous application refusal" in q
    ):
        return """
SELECT
    ROUND(AVG(PREV_REFUSAL_RATE) * 100, 2) AS average_refusal_rate
FROM applications
WHERE PREV_APPLICATION_COUNT > 0
"""

    return None


def llm_generate_sql(
    question: str,
    conversation=None,
) -> str:
    """
    Generic SQL generation. The model can answer questions outside the
    suggested examples because the complete schema and recent context are
    supplied dynamically.
    """

    context = build_conversation_context(
        conversation
    )

    if likely_follow_up(
        question,
        conversation,
    ):
        context_instruction = (
            "This question appears to be a follow-up. Resolve references "
            "such as 'that', 'it', 'them', 'what about', or 'compare' using "
            "the recent conversation. Preserve the earlier metric only when "
            "the wording clearly refers to it."
        )
    else:
        context_instruction = (
            "Treat this as a standalone question. Do not unnecessarily "
            "carry filters or metrics from previous turns."
        )

    prompt = f"""
{SQL_SCHEMA_PROMPT}

Recent conversation:
{context}

Current user question:
{question}

Context instruction:
{context_instruction}

Work through the user's intent internally, then return exactly ONE SQL
SELECT query that answers the question.

For broad or unusual questions:
- Select only the columns and aggregations needed to answer the question.
- You may combine multiple descriptive measures in one query when that
  provides better evidence for the requested explanation.
- If the user asks for a comparison, return the relevant groups side by side.
- If the user asks "why", query descriptive evidence that can support
  possible explanations from the available variables, without claiming
  causality that the data cannot establish.
- If the question is ambiguous and cannot be safely resolved from the
  conversation or schema, generate the safest useful query rather than
  inventing missing information.

Return SQL only.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": SQL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=700,
        reasoning_effort="low",
    )

    return clean_sql(
        response.choices[0].message.content
    )


def generate_sql(
    question: str,
    conversation=None,
) -> str:
    """
    Generate SQL for any supported question.

    Common questions use safe templates; everything else is handled by
    the LLM with schema and recent conversational context.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    # Common, deterministic queries first.
    sql = deterministic_sql(question)

    if sql:
        return clean_sql(sql)

    # Arbitrary and context-dependent questions go to the LLM.
    sql = llm_generate_sql(
        question,
        conversation=conversation,
    )

    if not sql:
        raise ValueError(
            "The SQL generator returned an empty query."
        )

    return clean_sql(sql)