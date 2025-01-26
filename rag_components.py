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

# %%
from openai import OpenAI
from markdown_processor import process_markdown_files
import psycopg2
import pandas as pd


# %%
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
            content TEXT,
            embedding VECTOR({embedding_dimension})
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
            INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s);
            """,
            (row["title"], row["content"], row["embedding"]),
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
    query = """SELECT * FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT 1;"""
    cursor.execute(query, (query_embedding,))
    result = cursor.fetchone()
    cursor.close()
    return result


# %%
client = OpenAI()

conn = psycopg2.connect(
    dbname="embedding_db",
    user="dev_user",
    password="dev_password",
    host="localhost",
    port="5433",
)

# %%

embeddings_df = process_markdown_files(client)

# %%

embedding_dimension = len(embeddings_df["embedding"][0])
print("Embedding dimension:", embedding_dimension)

# %%

set_up_embedding_table(conn, embedding_dimension)

# %%

insert_embeddings(conn, embeddings_df)

# %%

print_db_contents(conn)

# %%

query = "How do I use AI for my day-to-day tasks?"
result = query_vector_db(conn, client, query)
print(result)

# %%

conn.close()

# %%
