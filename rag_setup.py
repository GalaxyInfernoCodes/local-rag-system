from openai import OpenAI
import psycopg2
from markdown_processor import process_markdown_files
from rag_components import set_up_embedding_table, insert_embeddings, count_db_entries


client = OpenAI()

conn = psycopg2.connect(
    dbname="embedding_db",
    user="dev_user",
    password="dev_password",
    host="localhost",
    port="5433",
)


embeddings_df = process_markdown_files(client, "./markdown_source_files/")
embedding_dimension = len(embeddings_df["embedding"][0])
print("Embedding dimension:", embedding_dimension)

set_up_embedding_table(conn, embedding_dimension)
print("Table created")
insert_embeddings(conn, embeddings_df)
print("Embeddings inserted")

print("Database contains", count_db_entries(conn), "entries")

conn.close()
