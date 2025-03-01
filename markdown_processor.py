from openai import OpenAI
import pandas as pd
import glob
from markdown_it import MarkdownIt
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from tqdm import tqdm


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


def process_markdown_files(openai_client: OpenAI, config_path: str):
    # Load configuration from YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    markdown_sources = config.get("markdown_sources", [])

    client = openai_client

    md_contents = []
    for base_path in markdown_sources:
        # Use glob to find markdown files in subdirectories of each base_path
        md_files = glob.glob(f"{base_path}/**/*.md", recursive=True)
        md_contents.extend([extract_markdown_content(file) for file in md_files])

    print("Number of markdown files:", len(md_contents))

    md = MarkdownIt()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    data = []

    for content in tqdm(md_contents, desc="Processing markdown files"):
        headline, body = extract_markdown_info(md, content)

        # Split the body text into chunks
        split_contents = text_splitter.split_text(body)

        for chunk_id, chunk in enumerate(split_contents):
            # Replace summary with the first 50 words of the chunk
            first_50_words = " ".join(chunk.split()[:50])

            embedding_response = client.embeddings.create(
                input=chunk, model="text-embedding-3-large", dimensions=1600
            )
            embedding_vector = embedding_response.data[0].embedding

            data.append(
                {
                    "title": headline,
                    "start_of_chunk": first_50_words,
                    "embedding": embedding_vector,
                    "text_chunk": chunk,
                    "chunk_id": chunk_id,
                }
            )

    df = pd.DataFrame(data)
    return df
