import os
import sqlite3
import click
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data")
db_path = os.path.join(DB_DIR, "deals.db")

# Models
PARSE_MODEL = os.getenv("GPT_PARSE_MODEL", "gpt-4o")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            deal_name TEXT,
            entry_type TEXT,
            raw_note TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS deliverables (
            id INTEGER PRIMARY KEY,
            deal_name TEXT,
            description TEXT,
            due_date TEXT,
            agent TEXT,
            depends_on_id INTEGER,
            FOREIGN KEY(depends_on_id) REFERENCES deliverables(id),
            tags TEXT,
            metadata TEXT
        )
    ''')

    conn.commit()
    conn.close()

def store_journal_entry(deal_name, entry_type, raw_note, tags, metadata):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        'INSERT INTO journal (deal_name, entry_type, raw_note) VALUES (?, ?, ?)',
        (deal_name, entry_type, raw_note)
    )
    conn.commit()
    conn.close()

def store_deliverable(deal_name, description, due_date, agent, depends_on_desc=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    depends_on_id = None
    if depends_on_desc:
        c.execute(
            'SELECT id FROM deliverables WHERE deal_name=? AND description LIKE ?',
            (deal_name, f"%{depends_on_desc.strip()}%")
        )
        match = c.fetchone()
        if match:
            depends_on_id = match[0]
        else:
            click.secho(f"⚠️ Missing dependency: '{depends_on_desc}'", fg="yellow")
            create = click.confirm(f"Would you like to create the missing dependency '{depends_on_desc}'?")
            if create:
                c.execute('''
                    INSERT INTO deliverables (deal_name, description, due_date, agent, depends_on_id)
                    VALUES (?, ?, ?, ?, NULL)
                ''', (deal_name, depends_on_desc, None, None))
                depends_on_id = c.lastrowid
                click.secho(f"✅ Created dependency '{depends_on_desc}' with ID {depends_on_id}.", fg="green")

    c.execute('''
        INSERT INTO deliverables (deal_name, description, due_date, agent, depends_on_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (deal_name, description, due_date, agent, depends_on_id))

    conn.commit()
    conn.close()

def parse_and_distribute(entry_text: str) -> dict:
    function_schema = {
        "name": "extract_deal_metadata",
        "description": "Extract metadata from a free-text deal journal entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "deal_name": {"type": "string"},
                "entry_type": {"type": "string"},
                "notes": {"type": "string"},
                "deliverables": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dates": {
                    "type": "array",
                    "items": {"type": "string", "format": "date"}
                },
                "agents": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["deal_name", "entry_type", "notes"]
        }
    }

    messages = [
        {"role": "system", "content": "Extract deal_name, entry_type, notes, deliverables, dates, agents, and dependencies from the following journal entry."},
        {"role": "user", "content": entry_text}
    ]

    try:
        response = client.chat.completions.create(
            model=PARSE_MODEL,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "extract_deal_metadata"},
            temperature=0
        )
        return json.loads(response.choices[0].message.function_call.arguments)
    except Exception as e:
        click.secho(f"⚠️ AI parsing failed: {e}", fg="yellow")
        return {"raw_ai": entry_text}

@click.group()
def cli():
    init_db()

@cli.command()
@click.option('--text', prompt="Entry Text", help="Free text describing the journal entry.")
def log(text):
    metadata = parse_and_distribute(text)

    deal_name = metadata.get("deal_name")
    entry_type = metadata.get("entry_type")
    notes = metadata.get("notes")

    store_journal_entry(deal_name, entry_type, notes, tags="", metadata=metadata)

    deliverables = metadata.get("deliverables", [])
    dates = metadata.get("dates", [])
    agents = metadata.get("agents", [])
    dependencies = metadata.get("dependencies", [])

    for i, task in enumerate(deliverables):
        due = dates[i] if i < len(dates) else None
        agent = agents[i] if i < len(agents) else None
        depends_on = dependencies[i] if i < len(dependencies) else None
        store_deliverable(deal_name, task, due, agent, depends_on)

if __name__ == '__main__':
    cli()
