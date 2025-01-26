# ## The rough plan

# I want to build a local RAG system, using pgvector (from PostgreSQL) in a Docker container. In this I want to save the embeddings of my personal knowledge base, and be able to query it via prompts.

# 1. Start by creating embeddings from a few files and deciding on a data structure to store them.
# 2. Use the few embeddings to enrich a query to an LLM.
# 3. Set up a Docker container for the vector storage, including an index using pgvector.
# 4. Use the vector storage to query the knowledge base.
# %%
from openai import OpenAI
from markdown_processor import process_markdown_files
import psycopg2

client = OpenAI()

conn = psycopg2.connect(
    dbname="embedding_db",
    user="dev_user",
    password="dev_password",
    host="localhost",
    port="5433",
)

embeddings_df = process_markdown_files(client)

# %%

embedding_dimension = len(embeddings_df["embedding"][0])
print("Embedding dimension:", embedding_dimension)

# %%

# Open a cursor to perform database operations
cursor = conn.cursor()

# Execute SQL statements
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create the table with title, content, and embedding columns
cursor.execute(
    f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding VECTOR({embedding_dimension}) -- Adjust the dimension based on your actual embedding size
    );
    """
)

for index, row in embeddings_df.iterrows():
    cursor.execute(
        """
        INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s);
        """,
        (row["title"], row["content"], row["embedding"]),
    )

conn.commit()

# Fetch and print the results
cursor.execute("SELECT * FROM documents;")
items = cursor.fetchall()
for item in items:
    print(item)

# Close communication with the database
cursor.close()
conn.close()

# %%
