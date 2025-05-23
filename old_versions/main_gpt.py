
import openai
import pandas as pd
from datetime import datetime
import uuid
import json
from typing import List, Dict

# CONFIGURATION
openai.api_key = "your-api-key"  # Replace with your actual key
JOURNAL_PATH = "deal_journal.csv"

# Journal columns
journal_columns = [
    "Entry ID", "Date", "Deal Name", "Author", "Entry Type", "Raw Note", "AI Summary", "Tags"
]

# Function to get structured entries using OpenAI
def get_structured_logs_from_ai(raw_text: str, model="gpt-4") -> list:
    prompt = f"""
You are a deal assistant. Your job is to extract structured log entries from a loose private equity update.

Each entry should be a JSON object with these fields:
- Deal Name
- Entry Type (Meeting, Legal, Financial, DD, Note, etc.)
- Raw Note (keep the original wording)
- Tags (comma-separated, no hashtags)

Here is the unstructured input:
"""{raw_text}"""

Return a JSON array of log entries.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", content)
        raise e

# Function to add structured entries to the journal
def add_entries_to_journal(entries: List[Dict], journal_path: str = JOURNAL_PATH):
    try:
        journal_df = pd.read_csv(journal_path)
    except FileNotFoundError:
        journal_df = pd.DataFrame(columns=journal_columns)

    now = datetime.now().isoformat()
    for entry in entries:
        entry["Entry ID"] = str(uuid.uuid4())
        entry["Date"] = now
        entry["Author"] = "Ivan"
        entry["AI Summary"] = ""

    new_entries_df = pd.DataFrame(entries)
    updated_journal = pd.concat([journal_df, new_entries_df], ignore_index=True)
    updated_journal.to_csv(journal_path, index=False)
    print(f"Added {len(entries)} entries to {journal_path}")
    return updated_journal

# Example workflow
if __name__ == "__main__":
    print("Enter your update (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    raw_input_text = "\n".join(lines)

    structured_entries = get_structured_logs_from_ai(raw_input_text)
    updated_journal = add_entries_to_journal(structured_entries)
    print(updated_journal.tail())
