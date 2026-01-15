# GEMINI.md

## Project Overview

This project, `cortexindex`, is a local-only vault indexer and retriever. It's designed to be a Python package that can be used to index a local vault of files (like an Obsidian vault) and then retrieve information from it using a hybrid search (vector and lexical). The project is intended to be used by coding agents like Gemini, Claude, or Codex.

The project is built with Python and uses the following key technologies:

*   **Typer:** For the command-line interface.
*   **Sentence-Transformers:** For generating embeddings for vector search.
*   **SQLite:** As the backing store for the index, managed via the `libsql_store.py` module.
*   **Watchdog:** For monitoring file system events in watch mode.

The architecture is modular, with clear separation between the CLI, indexer, retriever, and data store. It supports various file types, including Markdown, PDF, PPTX, XLSX, and images. For images, it can use either Tesseract for OCR or the Gemini model for image analysis.

## Building and Running

To get started with this project, you need to have Python 3.11+ installed.

1.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -U pip
    pip install -e ".[dev]" # This will install google-genai
    ```

3.  **Initialize the configuration:**

    ```bash
    cortex init --vault "/path/to/your/vault" --index "~/.cortex/indexes/myvault"
    ```

4.  **Scan and index your vault:**

    ```bash
    cortex scan --full
    ```

5.  **Query your vault:**

    ```bash
    cortex query "your query"
    ```

### Running Tests

To run the tests, use `pytest`:

```bash
pytest
```

## Development Conventions

*   **Code Style:** The project uses `ruff` for linting and formatting.
*   **Typing:** The project uses type hints and `mypy` for static type checking.
*   **Modularity:** The codebase is organized into modules with specific responsibilities (e.g., `indexer`, `retrieval`, `store`).
*   **Configuration:** The project uses a `config.toml` file for configuration.
*   **Extensibility:** The use of registries for extractors and chunkers makes it easy to add support for new file types.

## Image Analysis Configuration

The image analysis feature can be configured in the `config.toml` file.

```toml
[image_analysis]
provider = "gemini" # or "tesseract"
```

There are two providers available:

*   `tesseract`: Uses the Tesseract OCR engine to extract text from images. This requires Tesseract to be installed on your system.
*   `gemini`: Uses the Google Gemini model to analyze images. This provides a richer description of the image content. To use this provider, you must set the `GEMINI_API_KEY` environment variable to your Google API key.

