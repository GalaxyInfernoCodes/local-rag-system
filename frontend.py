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
    result = query_vector_db(conn, client, message)
    if result:
        _, _, text_chunk, _, _ = result
        answer = answer_query_with_context(client, message, text_chunk)
    return answer


gr.ChatInterface(fn=chat_function, type="messages").launch()

conn.close()
