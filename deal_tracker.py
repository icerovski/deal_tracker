import os
import sqlite3
import click
import json
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
import shutil
import re
from datetime import datetime

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
            FOREIGN KEY(depends_on_id) REFERENCES deliverables(id)
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

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    init_db()
    if ctx.invoked_subcommand is None:
        choice = click.prompt("What would you like to do?", type=click.Choice(['log', 'schedule'], case_sensitive=False))
        if choice == 'log':
            ctx.invoke(log)
        elif choice == 'schedule':
            ctx.invoke(schedule)

@cli.command()
@click.option('--file', type=click.Path(exists=True), required=True, help="Text file containing multiple entries in format: Project {Name}:")
def batch_log(file):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    entries = re.split(r'(?=^Project\s+.+?:)', content, flags=re.MULTILINE)
    parsed_entries = [entry.strip() for entry in entries if entry.strip()]

    for entry_text in parsed_entries:
        match = re.match(r'^Project\s+(.+?):', entry_text)
        if not match:
            click.secho("❌ Could not find project name in entry:", fg="red")
            click.echo(entry_text)
            continue

        metadata = parse_and_distribute(entry_text)
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

    # Archive the file in /archive folder
    archive_dir = os.path.join(BASE_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archived_name = f"log_{timestamp}.txt"
    archive_path = os.path.join(archive_dir, archived_name)
    shutil.move(file, archive_path)
    click.secho(f"✅ Archived input file to {archive_path}", fg="green")

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

@cli.command()
@click.option('--all-projects', is_flag=True, help="Show deliverables for all projects instead of prompting.")
def schedule(all_projects):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()


    if all_projects:
        c.execute("SELECT deal_name, description, due_date FROM deliverables ORDER BY due_date")
        rows = c.fetchall()
        df = pd.DataFrame(rows, columns=["Project", "Task", "Due Date"])
        df.dropna(subset=["Due Date"], inplace=True)
        df["Due Date"] = pd.to_datetime(df["Due Date"])
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
        ax.barh(df["Task"], left=pd.to_datetime("today"), width=(df["Due Date"] - pd.to_datetime("today")).dt.days)
        ax.set_xlabel("Days from Today")
        ax.set_title("Gantt-like Chart of Deliverables")
        plt.tight_layout()
        plt.show()
    else:
        c.execute("SELECT DISTINCT deal_name FROM deliverables")
        projects = [row[0] for row in c.fetchall()]

        if not projects:
            click.echo("No projects found.")
            return

        project = click.prompt("Select a project", type=click.Choice(projects))

        c.execute("SELECT id, description, due_date, depends_on_id FROM deliverables WHERE deal_name=? ORDER BY due_date", (project,))
        rows = c.fetchall()

        dependency_lookup = {}
        c.execute("SELECT id, description FROM deliverables WHERE deal_name=?", (project,))
        for row in c.fetchall():
            dependency_lookup[row[0]] = row[1]

        if not rows:
            click.echo(f"No deliverables found for project {project}.")
            return

        click.secho(f"\n📅 Deliverables for {project}", fg="cyan")
        for due, desc, dep_id in [(r[2], r[1], r[3]) for r in rows]:
            dep_note = f"  ← depends on: {dependency_lookup.get(dep_id, 'Unknown')}" if dep_id else ""
            click.echo(f"  {due or 'No date'}: {desc}{dep_note}")

    conn.close()

if __name__ == '__main__':
    cli()
