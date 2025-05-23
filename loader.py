from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).parent / "pad.pdf"
CHUNK_KW = dict(chunk_size=1500, chunk_overlap=200)

def load_and_chunk(pdf_path: Path | str = PDF_PATH):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # Inject page number directly into the content
    for d in docs:
        if "page" in d.metadata:
            page = d.metadata["page"]
            d.page_content += f"\n\n(Source: PAD, p. {page})"

    return RecursiveCharacterTextSplitter(**CHUNK_KW).split_documents(docs)
