
import openai
import pandas as pd

def generate_deal_summary(deal_name: str, journal_path: str = "deal_journal.csv", model="gpt-4") -> str:
    try:
        journal_df = pd.read_csv(journal_path)
    except FileNotFoundError:
        return f"Journal file not found: {journal_path}"

    deal_entries = journal_df[journal_df["Deal Name"].str.lower() == deal_name.lower()]

    if deal_entries.empty:
        return f"No entries found for deal: {deal_name}"

    # Sort entries chronologically and format them
    note_history = "\n".join(
        f"- [{row['Date']}] {row['Raw Note']}" for _, row in deal_entries.iterrows()
    )

    prompt = f"""
You are a smart assistant that summarizes private equity deal developments based on a chronological activity journal.

Deal Name: {deal_name}

Here are the historical notes:

{note_history}

Please return a current summary of the deal in 3 parts:
1. Current overall status
2. Key developments and changes
3. Remaining risks and next steps

Be concise, factual, and helpful.
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
