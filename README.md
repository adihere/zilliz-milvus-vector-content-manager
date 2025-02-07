# zilliz-milvus-vecotr-database-manager
This Python Utility provides a Gradio interface for creating content on a Zilliz vector database. It offers functionality to add documents, process batches of files, and load content from URLs into the vector store.


# Zilliz Vector Database Manager

This Python application provides a Gradio interface for managing a Zilliz vector database. It offers functionality to add documents, process batches of files, and load content from URLs into the vector store.

## Features

- **Single Document Addition**: Add individual documents with a heading and content.
- **Batch Loading**: Process multiple text files from a specified directory.
- **URL Content Loading**: Fetch and process content from up to 10 URLs simultaneously.
- **Vector Embedding**: Uses SentenceTransformer for text embedding.
- **Logging**: Comprehensive logging of database updates and operations.

## Requirements

- Python 3.x
- Gradio
- LangChain
- SentenceTransformers
- Zilliz
- BeautifulSoup4
- Pandas
- Requests
- python-dotenv
- tqdm
- pymilvus

## Setup

1. Install the required dependencies:

```bash
pip install gradio langchain sentence-transformers zilliz beautifulsoup4 pandas requests python-dotenv tqdm pymilvus
```

2. Set up environment variables:
   - Create a `.env` file in the project root
   - Add the following variables:
     ```
     ZILLIZ_CLOUD_URI=your_zilliz_cloud_uri
     api_key_zilliz1=your_zilliz_api_key
     ```

3. Ensure you have an existing Zilliz collection named 'elevenplus'.

## Usage

Run the application:

```bash
python add-embeddings.py
```

The Gradio interface will be accessible via web browser, typically at `http://localhost:7960`.

## Key Components

- **Vector Store**: Utilizes Zilliz for storing and managing vector embeddings.
- **Text Splitting**: Implements RecursiveCharacterTextSplitter for chunking text.
- **Web Scraping**: Uses WebBaseLoader and BeautifulSoup for fetching and parsing web content.
- **Error Handling**: Robust error handling and logging mechanisms.

## Security Notes

- SSL verification is disabled by default (`VERIFY_SSL = False`). Set to `True` for production use.
- The application uses a custom user agent for web requests.
- Ensure proper read/write permissions for the application directory and log files.

## Logging

- Application logs are stored in the `logs` directory.


## Important Considerations

- The application is designed to work with an existing Zilliz collection named 'elevenplus'.
- It includes platform-specific configurations for different operating systems.
- The application attempts to create necessary directories with proper permissions.

## Contributing
Contributions to improve the application are welcome. Please ensure to follow best practices for code quality and security when submitting pull requests.



