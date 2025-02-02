from openai import OpenAI
import pandas as pd
import glob
from markdown_it import MarkdownIt
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_markdown_content(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def extract_markdown_info(md: MarkdownIt, content: str) -> tuple[str, str]:
    tokens = md.parse(content)
    # Extract the first headline
    headline = None
    for token in tokens:
        if token.type == "heading_open" and token.tag == "h1":
            # The next token should be the actual text of the heading
            headline = tokens[tokens.index(token) + 1].content
            break

    body = " ".join(token.content for token in tokens if token.type == "inline")
    return headline, body


def process_markdown_files(openai_client: OpenAI, base_path: str):
    client = openai_client

    # Use glob to find markdown files in subdirectories of the base_path
    md_files = glob.glob(f"{base_path}/**/*.md", recursive=True)
    md_contents = [extract_markdown_content(file) for file in md_files]

    md = MarkdownIt()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=8000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    data = []
    for content in md_contents:
        headline, body = extract_markdown_info(md, content)

        # Split the body text into chunks
        split_contents = text_splitter.split_text(body)

        for chunk_id, chunk in enumerate(split_contents):
            # Generate a short summary of each chunk using OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "developer",
                        "content": "Summarize the following document in a short sentence:",
                    },
                    {"role": "user", "content": chunk},
                ],
                max_tokens=50,
            )
            summary = completion.choices[0].message.content

            embedding_response = client.embeddings.create(
                input=chunk, model="text-embedding-3-large", dimensions=1600
            )
            embedding_vector = embedding_response.data[0].embedding

            data.append(
                {
                    "title": headline,
                    "summary": summary,
                    "embedding": embedding_vector,
                    "text_chunk": chunk,
                    "chunk_id": chunk_id,
                }
            )

    df = pd.DataFrame(data)
    return df
