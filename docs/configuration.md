# CorpusFlower Configuration

CorpusFlower uses environment variables (typically via a `.env` file loaded with `python-dotenv`) to configure models and paths.

## Core Settings

- `OPENAI_API_KEY`  
  Your OpenAI API key. Required for embeddings and synthesis.

- `OPENAI_MODEL`  
  The chat/completion model to use for synthesis (e.g., `gpt-4o-mini`).  
  If unset, a sensible default may be defined in the code.

- `OPENAI_EMBEDDING_MODEL`  
  The embedding model used for RootIndex (e.g., `text-embedding-3-small`).

## Optional Path Overrides

- `CORPUSFLOWER_PDF_PATH`  
  Overrides the default `indexer/pdfs/` directory for input PDFs.

- `CORPUSFLOWER_INDEX_PATH`  
  Directory where the vector index and any cache artifacts will be written.

## Example `.env`

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CORPUSFLOWER_PDF_PATH=./indexer/pdfs
CORPUSFLOWER_INDEX_PATH=./data/index
```

## Best Practices

- Never commit your `.env` file to version control.
- Document any new configuration flags you add as you extend the system.
- If you deploy CorpusFlower in a team environment, consider managing these settings via:
  - environment variables in your hosting platform, or
  - a secrets manager, rather than checked-in files.
