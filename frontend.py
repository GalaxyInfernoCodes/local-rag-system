import gradio as gr
from rag_components import answer_query_with_context, query_vector_db
from openai import OpenAI
import psycopg2

client = OpenAI()

conn = psycopg2.connect(
    dbname="embedding_db",
    user="dev_user",
    password="dev_password",
    host="localhost",
    port="5433",
)


def chat_function(message: str, history: str):
    # ignores the history for now - just use the last message
    results = query_vector_db(conn, client, message)
    if len(results) > 0:
        context = "\n".join(result[2] for result in results)  # Extracting text_chunks
        answer = answer_query_with_context(client, message, context)
    return answer


gr.ChatInterface(fn=chat_function, type="messages").launch()

conn.close()
