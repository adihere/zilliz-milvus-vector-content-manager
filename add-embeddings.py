import os
import logging
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Zilliz
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
import urllib3
import certifi
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin
import re

from tqdm import tqdm
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import connections, Collection, utility
import glob
from pathlib import Path
import sys
import platform

# Platform-specific configurations
PLATFORM = platform.system().lower()
HOME_DIR = Path.home()
APP_DIR = Path(__file__).parent.resolve()
LOGS_DIR = APP_DIR / 'logs'

# Application constants
DEFAULT_PORT = 7960
COLLECTION_NAME = 'elevenplus'  # Add collection name constant
ZILLIZ_CLOUD_URI = os.getenv('ZILLIZ_CLOUD_URI')
ZILLIZ_CLOUD_API_KEY = os.getenv('api_key_zilliz1')
CSV_LOGFILE = LOGS_DIR / 'loadedindb.csv'
CSV_COLUMNS = ['timestamp', 'heading', 'chunk_count', 'vector_id_start', 'vector_id_end']
VERIFY_SSL = False  # Set to True in production for security
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
REQUEST_TIMEOUT = 30

# Add new constants
MAX_CRAWL_DEPTH = 1
MAX_URLS_PER_DOMAIN = 50
IGNORED_EXTENSIONS = ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.xml']

# Ensure all required directories exist with proper permissions
def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist and verify permissions"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = path / '.write_test'
        test_file.touch()
        test_file.unlink()
        return path
    except (PermissionError, OSError) as e:
        # Fallback to user home directory if app directory is not writable
        fallback_dir = HOME_DIR / '.11plus' / path.name
        logger.warning(f"Cannot write to {path}, falling back to {fallback_dir}")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir

# Set up logging
LOGS_DIR = ensure_directory(LOGS_DIR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'vector_store.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_collection():
    """Get existing collection without recreating"""
    try:
        if not utility.has_collection(COLLECTION_NAME):
            raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist. Please create it first.")
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to collection: {str(e)}")
        raise

# Initialize Milvus connection
connections.connect(
    alias="default",
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_API_KEY,
    secure=True
)

# Get existing collection
collection = get_collection()

# Initialize vector store for searching
vector_store = Zilliz(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={
        "uri": ZILLIZ_CLOUD_URI,
        "token": ZILLIZ_CLOUD_API_KEY,
        "secure": True,
    }
)

def log_db_update(heading: str, chunk_count: int, start_id: int, end_id: int):
    timestamp = datetime.now().isoformat()
    data = {
        'timestamp': [timestamp],
        'heading': [heading],
        'chunk_count': [chunk_count],
        'vector_id_start': [start_id],
        'vector_id_end': [end_id]
    }
    df = pd.DataFrame(data)
    if not os.path.exists(CSV_LOGFILE):
        df.to_csv(CSV_LOGFILE, index=False, columns=CSV_COLUMNS)
    else:
        df.to_csv(CSV_LOGFILE, mode='a', header=False, index=False)

def split_text(text, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def add_to_vector_store(heading: str, text: str, progress=gr.Progress()):
    if not heading.strip():
        return "Error: Heading is required"
    
    try:
        chunks = split_text(text)
        progress(0, desc="Splitting text")
        
        timestamp = datetime.now().isoformat()
        metadata = [{"heading": heading, "timestamp": timestamp, "chunk_index": i} for i in range(len(chunks))]
        
        # Direct insertion using collection
        entities = []
        for i, chunk in enumerate(progress.tqdm(chunks, desc="Processing chunks")):
            vector = embeddings.embed_query(chunk)
            entities.append({
                "text": chunk,
                "vector": vector,
                "metadata": metadata[i]
            })
        
        progress(0, desc="Adding to vector store")
        mr = collection.insert(entities)
        
        # Get inserted IDs
        vector_id_start = min(mr.primary_keys)
        vector_id_end = max(mr.primary_keys)
        
        # Log update
        log_db_update(heading, len(chunks), vector_id_start, vector_id_end)
        
        progress(1, desc="Completed")
        return f"Added {len(chunks)} chunks (ID range: {vector_id_start}-{vector_id_end}) with heading '{heading}' at {timestamp}"
        
    except Exception as e:
        error_msg = f"Failed to add texts: {str(e)}"
        logger.error(error_msg)
        return f"Error adding texts to vector store: {str(e)}"

def read_file_content(file_path: str) -> str:
    """Read file content with fallback encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try binary read and decode with errors ignored
    try:
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path} with any encoding: {str(e)}")

def process_files_from_directory(directory: str, progress=gr.Progress()) -> str:
    """Process all text files from directory"""
    try:
        if not directory:
            return "Please select a directory"
        
        # Convert to Path object and resolve
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return f"Directory not found: {directory}"
        if not dir_path.is_dir():
            return f"Not a directory: {directory}"
            
        # Use Path.glob for cross-platform compatibility
        files = list(dir_path.glob("*.txt"))
        if not files:
            return "No .txt files found in directory"
            
        results = []
        for file_path in progress.tqdm(files, desc="Processing files"):
            try:
                if not os.access(file_path, os.R_OK):
                    results.append(f"Cannot read file {file_path}: Permission denied")
                    continue
                    
                content = read_file_content(str(file_path))
                heading = file_path.stem
                result = add_to_vector_store(heading, content, progress)
                results.append(f"File {heading}: {result}")
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                results.append(error_msg)
                
        return "\n\n".join(results)
        
    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def process_url(url: str, progress=gr.Progress()) -> str:
    """Process single URL and add content to vector store"""
    try:
        if not url.strip():
            return "Empty URL skipped"
            
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return f"Invalid URL format: {url}"
            
        progress(0, desc=f"Loading {url}")
        
        # Set up requests session with proper SSL handling
        session = requests.Session()
        if not VERIFY_SSL:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            # Use certifi for certificate verification
            session.verify = certifi.where()
            
        session.headers.update({
            'User-Agent': USER_AGENT
        })
        
        # Configure WebBaseLoader with session settings
        loader = WebBaseLoader(
            web_paths=[url],
            verify_ssl=VERIFY_SSL,
            requests_kwargs={
                'timeout': REQUEST_TIMEOUT,
                'verify': session.verify if VERIFY_SSL else False,
                'headers': session.headers
            }
        )
        
        try:
            docs = loader.load()
            # Use the same session for title extraction
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            heading = soup.title.string if soup.title else parsed.netloc
            
            # Process content using existing function
            result = add_to_vector_store(
                heading=heading,
                text=docs[0].page_content,
                progress=progress
            )
            
            return f"URL {url}: {result}"
            
        except requests.exceptions.RequestException as e:
            return f"Failed to fetch URL {url}: {str(e)}"
            
    except Exception as e:
        error_msg = f"Error processing URL {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def process_urls(*urls: str, progress=gr.Progress()) -> str:
    """Process multiple URLs
    Args:
        *urls: Variable number of URL strings
        progress: Gradio progress indicator
    """
    results = []
    valid_urls = [url for url in urls if url.strip()]
    
    if not valid_urls:
        return "No valid URLs provided"
        
    for url in valid_urls:
        result = process_url(url, progress)
        results.append(result)
    
    return "\n\n".join(results)

def is_valid_url(url: str, base_domain: str) -> bool:
    """Check if URL is valid and belongs to base domain"""
    try:
        parsed = urlparse(url)
        return (
            parsed.netloc.endswith(base_domain) and
            not any(url.lower().endswith(ext) for ext in IGNORED_EXTENSIONS)
        )
    except:
        return False

def get_robot_parser(base_url: str) -> RobotFileParser:
    """Initialize and return robots.txt parser"""
    rp = RobotFileParser()
    try:
        robots_url = urljoin(base_url, '/robots.txt')
        rp.set_url(robots_url)
        rp.read()
    except:
        pass
    return rp

def extract_links(soup: BeautifulSoup, base_url: str) -> set:
    """Extract valid links from page"""
    links = set()
    base_domain = urlparse(base_url).netloc.split('.')[-2:]
    base_domain = '.'.join(base_domain)
    
    for link in soup.find_all('a', href=True):
        url = urljoin(base_url, link['href'])
        if is_valid_url(url, base_domain):
            links.add(url)
    return links

async def crawl_website(base_url: str, progress=gr.Progress()) -> str:
    """Crawl website and add content to vector store"""
    try:
        processed_urls = set()
        results = []
        rp = get_robot_parser(base_url)
        
        if not rp.can_fetch("*", base_url):
            return "Access denied by robots.txt"
        
        session = requests.Session()
        if not VERIFY_SSL:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            session.verify = False
        session.headers.update({'User-Agent': USER_AGENT})
        
        # Process base URL first
        response = session.get(base_url, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = extract_links(soup, base_url)
        
        # Process base page
        heading = soup.title.string if soup.title else urlparse(base_url).netloc
        result = add_to_vector_store(heading, soup.get_text(), progress)
        results.append(f"Base URL: {result}")
        processed_urls.add(base_url)
        
        # Process linked pages
        for i, url in enumerate(links):
            if i >= MAX_URLS_PER_DOMAIN or url in processed_urls:
                continue
                
            try:
                progress(i / len(links), f"Processing {url}")
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                soup = BeautifulSoup(response.content, 'html.parser')
                heading = soup.title.string if soup.title else urlparse(url).netloc
                
                result = add_to_vector_store(heading, soup.get_text(), progress)
                results.append(f"Linked URL {url}: {result}")
                processed_urls.add(url)
                
            except Exception as e:
                results.append(f"Error processing {url}: {str(e)}")
                
        return "\n\n".join(results)
        
    except Exception as e:
        error_msg = f"Crawling failed for {base_url}: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Create tabbed interface
with gr.Blocks() as app:
    gr.Markdown("# Zilliz Vector Database Manager")
    
    with gr.Tabs():
        with gr.Tab("Single Document"):
            # Original interface
            gr.Interface(
                fn=add_to_vector_store,
                inputs=[
                    gr.Textbox(lines=1, label="Heading/Title", placeholder="Enter a descriptive heading"),
                    gr.Textbox(lines=10, label="Content", placeholder="Enter text to add to the vector store"),
                ],
                outputs=gr.Textbox(label="Result"),
                title="Add Single Document",
                description="Enter a single document to add to the database."
            )
            
        with gr.Tab("Batch Load"):
            with gr.Column():
                dir_input = gr.Textbox(
                    label="Directory Path",
                    placeholder="Enter directory path containing .txt files",
                    type="text"
                )
                process_btn = gr.Button("Process Directory", variant="primary")
                output = gr.Textbox(label="Processing Results", lines=10)
                
                process_btn.click(
                    fn=process_files_from_directory,
                    inputs=[dir_input],
                    outputs=[output]
                )
                
                gr.Markdown("""
                ### Instructions
                1. Enter the full path to a directory containing .txt files
                2. Each file should contain a single document
                3. Filenames will be used as document headings
                4. Click Process Directory to start batch loading
                """)
                
        with gr.Tab("URL Loader"):
            with gr.Column():
                url_inputs = [
                    gr.Textbox(
                        label=f"URL {i+1}",
                        placeholder="Enter webpage URL",
                        type="text"
                    ) for i in range(10)
                ]
                
                process_urls_btn = gr.Button("Process URLs", variant="primary")
                urls_output = gr.Textbox(label="Processing Results", lines=10)
                
                # Update click handler to use new function signature
                process_urls_btn.click(
                    fn=process_urls,
                    inputs=url_inputs,  # This will unpack the inputs as separate arguments
                    outputs=urls_output
                )
                
                gr.Markdown("""
                ### Instructions
                1. Enter up to 10 webpage URLs
                2. Each URL should point to a valid webpage
                3. Page titles will be used as document headings
                4. Click Process URLs to start loading content
                5. Empty URL fields will be skipped
                """)
                
        with gr.Tab("Website Crawler"):
            with gr.Column():
                website_url = gr.Textbox(
                    label="Website URL",
                    placeholder="Enter website URL to crawl",
                    type="text"
                )
                crawl_btn = gr.Button("Crawl Website", variant="primary")
                crawl_output = gr.Textbox(
                    label="Crawling Results", 
                    lines=15
                )
                
                crawl_btn.click(
                    fn=crawl_website,
                    inputs=[website_url],
                    outputs=[crawl_output]
                )
                
                gr.Markdown("""
                ### Website Crawler Instructions
                1. Enter the main URL of the website to crawl
                2. The crawler will process the main page and linked pages
                3. Maximum crawl depth: 1 level
                4. Maximum pages per domain: 50
                5. Respects robots.txt and only processes same-domain pages
                6. Skips non-HTML content (PDFs, images, etc.)
                """)

# Update launch parameters
if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', DEFAULT_PORT))
        app.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=port,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {str(e)}")
        sys.exit(1)
