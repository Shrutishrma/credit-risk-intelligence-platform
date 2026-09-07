from .nl_to_sql import generate_sql
from .sql_validator import validate_sql
from .query_runner import run_query
from .answer_generator import generate_business_answer


def ask_data(question, connection):
    """
    Complete NL -> SQL -> validation -> execution -> answer workflow.
    """

    # Generate SQL
    sql = generate_sql(question)

    # Validate SQL
    valid, message = validate_sql(sql)

    if not valid:
        return {
            "success": False,
            "message": message,
            "sql": sql
        }

    try:
        # Execute query
        result = run_query(connection, sql)

        # Generate business-readable answer
        answer = generate_business_answer(
            question,
            sql,
            result
        )

        return {
            "success": True,
            "sql": sql,
            "result": result,
            "answer": answer
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "sql": sql
        }