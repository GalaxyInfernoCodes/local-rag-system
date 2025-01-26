from openai import OpenAI
import pandas as pd
import glob
from markdown_it import MarkdownIt


def process_markdown_files(openai_client: OpenAI):
    client = openai_client

    md_files = glob.glob("*.md")
    md_contents = [open(file, "r").read() for file in md_files]

    md = MarkdownIt()

    data = []
    for content in md_contents:
        tokens = md.parse(content)
        # Extract the first headline
        headline = None
        for token in tokens:
            if token.type == "heading_open" and token.tag == "h1":
                # The next token should be the actual text of the heading
                headline = tokens[tokens.index(token) + 1].content
                break

        body = " ".join(token.content for token in tokens if token.type == "inline")

        # Generate a short summary of the whole document using OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": "Summarize the following document in a short sentence:",
                },
                {"role": "user", "content": body},
            ],
            max_tokens=50,
        )
        summary = completion.choices[0].message.content
        print("Summary:", summary)

        embedding_response = client.embeddings.create(
            input=body, model="text-embedding-3-large", dimensions=1600
        )
        embedding_vector = embedding_response.data[0].embedding

        data.append(
            {"title": headline, "content": summary, "embedding": embedding_vector}
        )

    df = pd.DataFrame(data)
    return df
