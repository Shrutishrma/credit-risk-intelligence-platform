import re

from .prompt_templates import ALLOWED_COLUMNS


FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "COPY",
    "EXPORT",
    "INSTALL",
    "PRAGMA"
]

ALLOWED_TABLES = {"applications"}

# Words that legitimately appear in uppercase inside generated SQL but are
# not applicant data columns, so they must not trip the column whitelist.
_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
    "AS", "AND", "OR", "NOT", "IN", "IS", "NULL", "DESC", "ASC", "COUNT",
    "AVG", "SUM", "MIN", "MAX", "ROUND", "DISTINCT", "ON", "JOIN", "CASE",
    "WHEN", "THEN", "ELSE", "END", "BETWEEN", "LIKE", "APPLICATIONS",
}

_ALLOWED_COLUMNS_UPPER = {c.upper() for c in ALLOWED_COLUMNS}


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate generated SQL before execution.
    Only read-only SELECT queries on the applications table are allowed.
    """

    if not sql or not sql.strip():
        return False, "Empty SQL query."

    sql_clean = sql.strip()

    # Remove SQL comments
    sql_clean = re.sub(
        r"--.*?$",
        "",
        sql_clean,
        flags=re.MULTILINE
    )

    sql_clean = re.sub(
        r"/\*.*?\*/",
        "",
        sql_clean,
        flags=re.DOTALL
    ).strip()

    # Must start with SELECT
    if not re.match(r"^SELECT\b", sql_clean, re.IGNORECASE):
        return False, "Only SELECT queries are allowed."

    # Block dangerous SQL operations
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(
            rf"\b{keyword}\b",
            sql_clean,
            re.IGNORECASE
        ):
            return False, f"Forbidden SQL operation: {keyword}"

    # Find tables used in FROM and JOIN
    tables = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql_clean,
        re.IGNORECASE
    )

    # If SQL contains FROM/JOIN, every table must be allowed
    for table in tables:
        if table.lower() not in {
            allowed.lower()
            for allowed in ALLOWED_TABLES
        }:
            return False, f"Unauthorized table: {table}"

    # Analytical queries should reference our table
    if not tables:
        return False, "No valid table reference found."

    # Block SELECT * outright - the whitelist below can't stop a wildcard,
    # and it's the direct route to leaking columns like SK_ID_CURR.
    if re.search(r"SELECT\s+\*", sql_clean, re.IGNORECASE) or re.search(
        r"SELECT\s+\w+\.\*", sql_clean, re.IGNORECASE
    ):
        return False, "SELECT * is not allowed; only whitelisted columns may be returned."

    # Column whitelist - heuristic, not a full SQL parser: every ALL-CAPS
    # identifier in the query must be a known column, a SQL keyword, or the
    # table name. Catches the common leak cases (e.g. a stray SK_ID_CURR)
    # without needing to parse the SELECT clause precisely.
    identifiers = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", sql_clean))
    unknown = identifiers - _SQL_KEYWORDS - _ALLOWED_COLUMNS_UPPER - {"SELECT"}

    for keyword in FORBIDDEN_KEYWORDS:
        unknown.discard(keyword)

    if unknown:
        return False, f"Query references unauthorized column(s): {', '.join(sorted(unknown))}"

    return True, "Valid SQL."