import sys
from pathlib import Path

# Add the project root to Python's path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import pymupdf
from config import RAW_DIR, PROCESSED_DIR

def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text page by page, returning a list of dicts."""
    doc = pymupdf.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        
        # Skip completely empty pages (common in huge legal docs)
        if not text.strip():
            continue
            
        pages.append({
            "page_num": page_num,
            "text": text.strip(),
            "char_count": len(text)
        })
        
    return pages

def main():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in data/raw/")
        return
        
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pdf_files:
        print(f"📄 Processing: {pdf_path.name}")
        start_time = time.time()
        
        pages = extract_text(pdf_path)
        
        # Save as JSON instead of flat TXT
        out_path = PROCESSED_DIR / f"{pdf_path.stem}_pages.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
            
        elapsed = time.time() - start_time
        total_chars = sum(p["char_count"] for p in pages)
        
        print(f"  ✅ Extracted {len(pages)} pages")
        print(f"  📝 Total characters: {total_chars:,}")
        print(f"  ⏱️ Time: {elapsed:.1f}s")
        print(f"  💾 Saved to: {out_path}\n")

if __name__ == "__main__":
    main()