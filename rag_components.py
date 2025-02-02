# ## The rough plan

# I want to build a local RAG system, using pgvector (from PostgreSQL) in a Docker container. In this I want to save the embeddings of my personal knowledge base, and be able to query it via prompts.

# 1. Start by creating embeddings from a few files and deciding on a data structure to store them.
# 2. Use the few embeddings to enrich a query to an LLM.
# 3. Set up a Docker container for the vector storage, including an index using pgvector.
# 4. Use the vector storage to query the knowledge base.

## Todo list
# - add the file name/path to the dataframe and table
# - write function to retrieve the file contents for the most similar embedding
# - write function to enrich the query with the found content
# - generate answer based on query and found context
# - check chunking - right now the embedding inputs might actually be too long and get truncated, so we need to chunk them
# - implement the chunking

from openai import OpenAI
import psycopg2
import pandas as pd


def set_up_embedding_table(
    conn: psycopg2.extensions.connection, embedding_dimension: int = 1600
):
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("DROP TABLE IF EXISTS documents;")

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            summary TEXT,
            embedding VECTOR({embedding_dimension}),
            text_chunk TEXT,
            chunk_id INTEGER
        );
        """
    )
    conn.commit()
    cursor.close()


def insert_embeddings(
    conn: psycopg2.extensions.connection, embeddings_df: pd.DataFrame
):
    cursor = conn.cursor()

    for _, row in embeddings_df.iterrows():
        cursor.execute(
            """
            INSERT INTO documents (title, summary, embedding, text_chunk, chunk_id) VALUES (%s, %s, %s, %s, %s);
            """,
            (
                row["title"],
                row["summary"],
                row["embedding"],
                row["text_chunk"],
                row["chunk_id"],
            ),
        )

    conn.commit()
    cursor.close()


def print_db_contents(conn: psycopg2.extensions.connection):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    cursor.close()


def count_db_entries(conn: psycopg2.extensions.connection) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents;")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def create_embedding_for_string(open_ai_client: OpenAI, input_str: str) -> list[float]:
    embedding_response = open_ai_client.embeddings.create(
        input=input_str, model="text-embedding-3-large", dimensions=1600
    )
    embedding_vector = embedding_response.data[0].embedding
    return embedding_vector


def query_vector_db(
    conn: psycopg2.extensions.connection, open_ai_client: OpenAI, query: str
):
    query_embedding = create_embedding_for_string(open_ai_client, query)

    cursor = conn.cursor()
    query = """SELECT title, summary, text_chunk, chunk_id, embedding FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT 1;"""
    cursor.execute(query, (query_embedding,))
    result = cursor.fetchone()
    cursor.close()

    if result:
        title, summary, text_chunk, chunk_id, embedding = result
        print(f"Most similar document: {title}")
        print(f"Summary: {summary}")
        print(f"Text Chunk: {text_chunk}")
        print(f"Chunk ID: {chunk_id}")

    return result


def answer_query_with_context(
    open_ai_client: OpenAI, user_query: str, context: str
) -> str:
    # Send the user's query along with the context to the LLM
    completion = open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Use the provided context to answer the user's query.",
            },
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": context},
        ],
        max_tokens=2000,
    )
    answer = completion.choices[0].message.content
    return answer
