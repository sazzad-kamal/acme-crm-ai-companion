"""
Markdown document ingestion and chunking for Acme CRM docs.

Responsibilities:
- Walk data/docs/ and load all *.md files
- Use heading-aware chunking (split by ## / ### then recursive character split)
- Target chunk size: 400-700 tokens with small overlap
- Save chunks to data/processed/doc_chunks.parquet

Usage:
    python -m project1_rag.ingest_docs
"""

import re
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from project1_rag.doc_models import DocumentChunk


# =============================================================================
# Configuration
# =============================================================================

DOCS_DIR = Path("data/docs")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "doc_chunks.parquet"

# Chunking parameters
TARGET_CHUNK_SIZE = 500  # tokens (approximate)
MAX_CHUNK_SIZE = 700
MIN_CHUNK_SIZE = 100
CHUNK_OVERLAP = 50  # tokens overlap between chunks

# Approximate tokens per character (for English text)
CHARS_PER_TOKEN = 4


# =============================================================================
# Markdown Parsing
# =============================================================================

def extract_title(content: str, filename: str) -> str:
    """Extract the document title from first H1 heading or use filename."""
    # Look for # Title at the start
    match = re.match(r'^#\s+(.+?)(?:\n|$)', content.strip())
    if match:
        return match.group(1).strip()
    return filename.replace('_', ' ').replace('-', ' ').title()


def split_by_headings(content: str) -> list[dict]:
    """
    Split markdown content by headings (##, ###, etc.).
    
    Returns a list of dicts with:
        - section_path: list of heading hierarchy
        - text: the section content
        - level: heading level (2 for ##, 3 for ###, etc.)
    """
    # Pattern to match headings (## or ### or ####)
    heading_pattern = re.compile(r'^(#{2,4})\s+(.+?)$', re.MULTILINE)
    
    sections = []
    current_path = []
    last_end = 0
    
    # Find the document title (H1) if present
    title_match = re.match(r'^#\s+(.+?)(?:\n|$)', content.strip())
    doc_start = 0
    if title_match:
        doc_start = title_match.end()
    
    # Find all headings
    matches = list(heading_pattern.finditer(content))
    
    if not matches:
        # No headings found, treat entire content as one section
        text = content[doc_start:].strip()
        if text:
            sections.append({
                "section_path": [],
                "text": text,
                "level": 0
            })
        return sections
    
    # Process content before first heading
    pre_heading_text = content[doc_start:matches[0].start()].strip()
    if pre_heading_text:
        sections.append({
            "section_path": ["Introduction"],
            "text": pre_heading_text,
            "level": 1
        })
    
    # Process each heading and its content
    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # characters
        heading_text = match.group(2).strip()
        
        # Update section path based on level
        # Level 2 (##) resets path, level 3 (###) appends, etc.
        if level == 2:
            current_path = [heading_text]
        elif level == 3:
            current_path = current_path[:1] + [heading_text]
        elif level == 4:
            current_path = current_path[:2] + [heading_text]
        
        # Get content until next heading or end
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        
        if text:
            sections.append({
                "section_path": current_path.copy(),
                "text": text,
                "level": level
            })
    
    return sections


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def recursive_split(text: str, max_size: int, overlap: int) -> list[str]:
    """
    Recursively split text into chunks of approximately max_size tokens.
    
    Tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Words (spaces)
    """
    estimated_tokens = estimate_tokens(text)
    
    if estimated_tokens <= max_size:
        return [text]
    
    # Try splitting by paragraphs first
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for part in parts:
                test_chunk = current_chunk + sep + part if current_chunk else part
                if estimate_tokens(test_chunk) <= max_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(chunks) > 1:
                # Add overlap between chunks
                result = []
                for i, chunk in enumerate(chunks):
                    if i > 0 and overlap > 0:
                        # Add some content from the end of previous chunk
                        prev_words = chunks[i-1].split()[-overlap:]
                        chunk = " ".join(prev_words) + " " + chunk
                    result.append(chunk.strip())
                return result
    
    # If nothing worked, just split by character count
    char_limit = max_size * CHARS_PER_TOKEN
    chunks = []
    for i in range(0, len(text), char_limit - overlap * CHARS_PER_TOKEN):
        chunks.append(text[i:i + char_limit])
    return chunks


# =============================================================================
# Document Processing
# =============================================================================

def process_markdown_file(file_path: Path) -> list[DocumentChunk]:
    """
    Process a single markdown file into DocumentChunks.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        List of DocumentChunk objects
    """
    content = file_path.read_text(encoding="utf-8")
    doc_id = file_path.stem  # filename without extension
    title = extract_title(content, doc_id)
    
    # Split by headings
    sections = split_by_headings(content)
    
    chunks = []
    chunk_index = 0
    
    for section in sections:
        section_text = section["text"]
        section_path = section["section_path"]
        
        # Check if section needs further splitting
        if estimate_tokens(section_text) > MAX_CHUNK_SIZE:
            sub_chunks = recursive_split(section_text, TARGET_CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            sub_chunks = [section_text]
        
        for sub_chunk in sub_chunks:
            # Skip very small chunks
            if estimate_tokens(sub_chunk) < MIN_CHUNK_SIZE // 2:
                continue
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}::{chunk_index}",
                doc_id=doc_id,
                title=title,
                text=sub_chunk,
                metadata={
                    "file_name": file_path.name,
                    "section_path": section_path,
                    "section_heading": section_path[-1] if section_path else None,
                    "chunk_index": chunk_index,
                    "estimated_tokens": estimate_tokens(sub_chunk),
                }
            )
            chunks.append(chunk)
            chunk_index += 1
    
    return chunks


def ingest_all_docs(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]:
    """
    Ingest all markdown files from the docs directory.
    
    Args:
        docs_dir: Path to the docs directory
        
    Returns:
        List of all DocumentChunks
    """
    all_chunks = []
    md_files = sorted(docs_dir.glob("*.md"))
    
    print(f"Found {len(md_files)} markdown files in {docs_dir}")
    
    for file_path in md_files:
        print(f"  Processing: {file_path.name}")
        chunks = process_markdown_file(file_path)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks")
    
    return all_chunks


def save_chunks(chunks: list[DocumentChunk], output_path: Path = OUTPUT_FILE) -> None:
    """
    Save chunks to a Parquet file.
    
    Args:
        chunks: List of DocumentChunk objects
        output_path: Path to the output Parquet file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    records = []
    for chunk in chunks:
        record = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": json.dumps(chunk.metadata),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path = OUTPUT_FILE) -> list[DocumentChunk]:
    """
    Load chunks from a Parquet file.
    
    Args:
        input_path: Path to the Parquet file
        
    Returns:
        List of DocumentChunk objects
    """
    df = pd.read_parquet(input_path)
    
    chunks = []
    for _, row in df.iterrows():
        chunk = DocumentChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            title=row["title"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
        )
        chunks.append(chunk)
    
    return chunks


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """Main entrypoint for document ingestion."""
    print("=" * 60)
    print("Acme CRM Docs Ingestion")
    print("=" * 60)
    
    # Ingest all docs
    chunks = ingest_all_docs()
    
    # Calculate stats
    total_tokens = sum(c.metadata.get("estimated_tokens", 0) for c in chunks)
    unique_docs = len(set(c.doc_id for c in chunks))
    
    print()
    print("Summary:")
    print(f"  - Loaded {unique_docs} docs")
    print(f"  - Produced {len(chunks)} chunks")
    print(f"  - Total estimated tokens: {total_tokens:,}")
    print(f"  - Avg tokens per chunk: {total_tokens // len(chunks) if chunks else 0}")
    
    # Save chunks
    save_chunks(chunks)
    print()
    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
