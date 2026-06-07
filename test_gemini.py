"""Simple Gemini API tester for the SciFind backend.

Usage:
    python test_gemini.py --question "Apa rekomendasi film sci-fi seperti Dune?" --title Dune

This script sends a POST request to http://localhost:5000/api/gemini and prints the JSON response.
"""

import argparse
import json
import os
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description="Test the Gemini API endpoint.")
    parser.add_argument("--question", required=True, help="Question to ask Gemini")
    parser.add_argument("--title", default="", help="Optional movie title context")
    parser.add_argument("--url", default=os.getenv("GEMINI_API_URL", "http://localhost:5000/api/gemini"), help="Gemini endpoint URL")
    args = parser.parse_args()

    payload = {
        "question": args.question,
        "title": args.title,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_text = resp.read().decode("utf-8")
            parsed = json.loads(response_text)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
    except json.JSONDecodeError:
        print("Failed to parse response as JSON.")


if __name__ == "__main__":
    main()
