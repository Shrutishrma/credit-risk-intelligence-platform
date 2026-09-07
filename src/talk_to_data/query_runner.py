import duckdb
import pandas as pd


def run_query(
    connection: duckdb.DuckDBPyConnection,
    sql: str
) -> pd.DataFrame:
    """
    Execute a validated SQL query and return the result.
    """

    return connection.execute(sql).fetchdf()