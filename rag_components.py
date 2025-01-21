# ## The rough plan

# I want to build a local RAG system, using pgvector (from PostgreSQL) in a Docker container. In this I want to save the embeddings of my personal knowledge base, and be able to query it via prompts.

# 1. Start by creating embeddings from a few files and deciding on a data structure to store them.
# 2. Use the few embeddings to enrich a query to an LLM.
# 3. Set up a Docker container for the vector storage, including an index using pgvector.
# 4. Use the vector storage to query the knowledge base.

from openai import OpenAI
from markdown_processor import process_markdown_files
import psycopg2

client = OpenAI()

process_markdown_files(client)

conn = psycopg2.connect(
    dbname="embedding_db",
    user="dev_user",
    password="dev_password",
    host="localhost",
    port="5433",
)

# Open a cursor to perform database operations
cursor = conn.cursor()

# Execute SQL statements
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS items (
        id SERIAL PRIMARY KEY,
        embedding VECTOR(3) -- Example for 3-dimensional vectors
    );
"""
)

cursor.execute(
    """
    INSERT INTO items (embedding) VALUES 
    ('[0.1, 0.2, 0.3]'), 
    ('[0.4, 0.5, 0.6]');
"""
)

# Fetch and print the results
cursor.execute("SELECT * FROM items;")
items = cursor.fetchall()
for item in items:
    print(item)

# Close communication with the database
cursor.close()
conn.close()
