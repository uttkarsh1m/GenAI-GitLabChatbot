"""
Data Ingestion Script
Run this once to scrape GitLab pages and build the vector index.
Usage: python ingest.py
"""

import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("  GitBot - Data Ingestion Pipeline")
    print("=" * 60)

    try:
        # Step 1: Scrape GitLab pages
        print("\n📡 Step 1/3: Scraping GitLab Handbook & Direction pages...")
        from src.scraper import scrape_all_sources

        start = time.time()
        documents = scrape_all_sources("data/gitlab_content.json")
        elapsed = time.time() - start

        if not documents:
            logger.error("❌ No documents scraped! Check network connectivity.")
            sys.exit(1)

        print(f"✅ Scraped {len(documents)} pages in {elapsed:.1f}s")

        # Step 2: Create chunks
        print("\n✂️  Step 2/3: Creating text chunks...")
        from src.chunker import create_chunks_from_documents, save_chunks, get_chunk_stats

        chunks = create_chunks_from_documents(documents)
        if not chunks:
            logger.error("❌ No chunks created! Check document content.")
            sys.exit(1)

        success = save_chunks(chunks, "data/chunks.json")
        if not success:
            logger.error("❌ Failed to save chunks!")
            sys.exit(1)

        stats = get_chunk_stats(chunks)
        print(f"✅ Created {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
        print(f"   Average chunk size: {stats['avg_words']:.0f} words")
        print(f"   By category: {stats['by_category']}")

        # Step 3: Build vector index
        print("\n🔢 Step 3/3: Building FAISS vector index...")
        print("   (This requires downloading the embedding model on first run ~90MB)")
        from src.embeddings import build_index

        start = time.time()
        index, _ = build_index(chunks, "data/faiss_index", "data/chunks.json")
        elapsed = time.time() - start

        if index is None:
            logger.error("❌ Failed to build index!")
            sys.exit(1)

        print(f"✅ Built index with {index.ntotal} vectors in {elapsed:.1f}s")

        print("\n" + "=" * 60)
        print("  ✅ Data ingestion complete!")
        print("  Run: streamlit run app.py")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
