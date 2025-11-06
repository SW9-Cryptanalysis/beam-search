import os
import requests
from utils.formatting import format_text

GUTENDEX_BASE_URL = "https://gutendex.com/books"
BOOKS_DIR = "books"


class Fetcher:
    """Fetch and save formatted Project Gutenberg books for training."""

    BOOK_IDS = [
        "84", "2701", "1342", "2641", "145", "37106", "7241", "67979", "43", "1260",
        "16389", "394", "6761", "345", "1259", "2160", "4085", "5197", "6593", "1232",
        "3207", "2554", "1080", "174", "98", "25344", "7370", "2148", "76", "1952",
        "2591", "2600", "41", "844", "46", "1661", "3296", "408", "5200", "26184",
        "205", "1497", "1998", "23", "768", "28054", "2542", "45", "34901", "219",
        "20203", "76939", "1184", "15399", "1400", "74", "36034", "815", "4300",
        "1023", "4363", "2852", "34450", "36", "55", "3300", "135", "2680", "829",
        "120", "12", "16", "60976", "140", "1399", "56517", "52621", "1228", "18269",
        "2814", "10554", "10007", "33944", "11", "236", "4351", "64317", "8438", "26659",
    ]

    def __init__(self) -> None:
        os.makedirs(BOOKS_DIR, exist_ok=True)

    def fetch_all_books(self) -> None:
        """Fetch, format, and save each book to the books/ folder."""
        for book_id in self.BOOK_IDS:
            path = os.path.join(BOOKS_DIR, f"{book_id}.txt")

            # Skip already-downloaded books
            if os.path.exists(path):
                print(f"✅ {book_id} already saved, skipping.")
                continue

            try:
                print(f"📚 Fetching book {book_id}...")
                metadata = requests.get(f"{GUTENDEX_BASE_URL}/{book_id}", timeout=10).json()
                formats = metadata.get("formats", {})

                # Try to find a usable plain text link
                text_url = next(
                    (
                        formats[k]
                        for k in [
                            "text/plain; charset=utf-8",
                            "text/plain; charset=us-ascii",
                            "text/plain",
                        ]
                        if k in formats
                    ),
                    None,
                )

                if not text_url:
                    print(f"⚠️ No plain text format for book {book_id}, skipping.")
                    continue

                # Fetch the book text
                text_response = requests.get(text_url, timeout=10)
                text_response.raise_for_status()

                # Format and save
                formatted = format_text(text_response.text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(formatted)

                print(f"💾 Saved formatted book {book_id} to {path}")

            except Exception as e:
                print(f"❌ Failed to fetch book {book_id}: {e}")

if __name__ == "__main__":
    fetcher = Fetcher()
    fetcher.fetch_all_books()