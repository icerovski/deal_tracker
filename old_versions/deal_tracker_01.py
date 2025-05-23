import os
import sqlite3
import click
from openai import OpenAI, RateLimitError
import json
import time
from dotenv import load_dotenv

# Path constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data")
db_path = os.path.join(DB_DIR, "deals.db")

# Recommended GPT models (using accessible 3.5 models)
PARSE_MODEL = os.getenv("GPT_PARSE_MODEL", "gpt-4o")
SUMMARY_MODEL = os.getenv("GPT_SUMMARY_MODEL", "gpt-4o")

# Retry settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# Instantiate OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def init_db():
    """
    Initialize the database and ensure required tables exist.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Free-form journal
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            deal_name TEXT,
            entry_type TEXT,
            raw_note TEXT
        )
    ''')

    # Structured deliverables
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


def call_ai_with_retry(model, messages):
    """
    Call the OpenAI chat completion endpoint with retries on rate limit.
    """
    delay = 1
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
        except RateLimitError:
            if attempt < MAX_RETRIES:
                click.secho(f"⚠️ Rate limit hit (attempt {attempt}/{MAX_RETRIES}), retrying in {delay}s...", fg="yellow")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                click.secho("⚠️ Rate limit exceeded after retries. Please try again later.", fg="red")
                return None


def parse_and_distribute(entry_text: str) -> dict:
    """
    Parse a journal entry using OpenAI function calling for structured output.
    """
    function_schema = {
    "name": "extract_deal_metadata",
    "description": "Extract structured deal and scheduling information from a journal entry.",
    "parameters": {
        "type": "object",
        "properties": {
            "deal_name": {"type": "string"},
            "entry_type": {"type": "string"},
            "notes": {"type": "string"},
            "deliverables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key actions or deliverables mentioned."
            },
            "dates": {
                "type": "array",
                "items": {"type": "string", "format": "date"},
                "description": "Relevant dates (deadlines, meetings, milestones) in ISO format (YYYY-MM-DD)."
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tasks or events this entry depends on."
            }
        },
        "required": ["deal_name", "entry_type", "notes"]
    }
}

    messages = [
        {
            "role": "system",
            "content": "You extract structured metadata from deal journal entries using function calling."
        },
        {
            "role": "user",
            "content": entry_text
        }
    ]

    try:
        response = client.chat.completions.create(
            model=PARSE_MODEL,
            messages=messages,
            functions=[function_schema],
            function_call={"name": "extract_journal_entry_metadata"},
            temperature=0
        )
        arguments = response.choices[0].message.function_call.arguments
        return json.loads(arguments)
    except Exception as e:
        click.secho(f"⚠️ Function call failed: {e}", fg="yellow")
        return {"raw_ai": entry_text}

# def parse_and_distribute(entry_text: str) -> dict:
#     """
#     Parse a free-text journal entry via AI and return structured metadata.
#     """

#     system_prompt = (
#     "You are an assistant that extracts structured metadata from journal entries. "
#     "Return only valid JSON. Do not include explanations, markdown, or formatting. "
#     "The JSON must contain keys: deal_id (string), deal_name (string), amount (number), "
#     "stage (string), contact (string), entry_type (string), tags (list of strings), notes (string). "
#     "If a field is missing, use null. Respond ONLY with raw JSON."
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": entry_text}
#     ]
#     response = call_ai_with_retry(PARSE_MODEL, messages)
#     if response is None:
#         return {"raw_ai_error": "rate_limit_exceeded"}
#     content = response.choices[0].message.content.strip()
#     try:
#         return json.loads(content)
#     except json.JSONDecodeError:
#         click.secho("⚠️ Failed to parse AI response as JSON; storing raw AI output.", fg="yellow")
#         return {"raw_ai": content}

def store_deliverable(deal_name, description, due_date, agent, depends_on_desc=None):
    """
    Save a deliverable and resolve its dependency by description.
    Warn if dependency is not found.
    """
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
            click.secho(f"⚠️ Missing dependency: '{depends_on_desc}' — please create this deliverable manually.", fg="yellow")

    c.execute('''
        INSERT INTO deliverables (deal_name, description, due_date, agent, depends_on_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (deal_name, description, due_date, agent, depends_on_id))

    conn.commit()
    conn.close()

def store_journal_entry(deal_name: str, entry_type: str, raw_note: str, tags: str, metadata: dict):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        'INSERT INTO journal (deal_name, entry_type, raw_note, tags, metadata) VALUES (?, ?, ?, ?, ?)',
        (deal_name, entry_type, raw_note, tags, json.dumps(metadata))
    )
    conn.commit()
    conn.close()


def store_deal_summary(deal_name: str, summary_text: str, metadata: dict):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        'INSERT INTO summaries (deal_name, summary_text, metadata) VALUES (?, ?, ?)',
        (deal_name, summary_text, json.dumps(metadata))
    )
    conn.commit()
    conn.close()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    init_db()
    if ctx.invoked_subcommand is None:
        choice = click.prompt("What would you like to do? (log/summarize)", type=click.Choice(['log','summarize']))
        ctx.invoke(log if choice=='log' else summarize)

@cli.command()
@click.option('--batch-file', type=click.Path(exists=True), required=False,
              help="Optional. If provided, logs entries in format: Project Name [Type]: Entry #tags.")
def log(batch_file):
    import re

    allowed_entry_types = {"meeting", "legal", "financial", "dd", "note"}
    pattern = r"Project\s+(.+?)\s*\[(\w+)\]:\s*(.+?)(?=(?:\nProject\s|$))"

    if not batch_file:
        draft_path = os.path.join(os.getcwd(), "draft.txt")

        template = (
            "# Write each entry on a new line using the format:\n"
            "# Project {Project_Name} [EntryType]: Entry text with optional #tags\n"
            "#\n"
            f"# Valid entry types: {', '.join(t.title() for t in allowed_entry_types)}\n"
            "# Example:\n"
            "# Project Titan [Meeting]: Met with EDF to discuss the new timeline. #grid #PPA\n"
            "#\n"
            "# After you save and close this file, the entries will be automatically logged to the database and this file will be deleted.\n\n"
        )

        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(template)

        click.edit(filename=draft_path)

        with open(draft_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            click.echo("No entries provided. Aborting.")
            return

        batch_file = draft_path

    with open(batch_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Filter out comment lines
    content = "\n".join(line for line in content.splitlines() if not line.strip().startswith("#"))

    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        click.echo("❌ No valid entries found. Use format: Project {Name} [Type]: Entry #tags")
        return

    valid_entries = 0
    for idx, (deal_name, entry_type, raw_note) in enumerate(matches, start=1):
        entry_type_normalized = entry_type.lower().strip()
        raw_note = raw_note.strip()

        if entry_type_normalized not in allowed_entry_types:
            click.secho(f"⚠️ Skipping entry {idx}: Invalid entry type '{entry_type}'. Allowed: {sorted(allowed_entry_types)}", fg="yellow")
            continue

        inline_tags = re.findall(r"#(\w+)", raw_note)
        cleaned_note = re.sub(r"#\w+", "", raw_note).strip()
        tags = ",".join(sorted(set(tag.lower() for tag in inline_tags)))

        click.echo(f"\n📌 Entry {idx} → Project: '{deal_name}', Type: '{entry_type.title()}', Tags: {tags or 'None'}")
        metadata = parse_and_distribute(cleaned_note)
        store_journal_entry(deal_name, entry_type.title(), cleaned_note, tags, metadata)
        valid_entries += 1

    if valid_entries > 0:
        os.remove(batch_file)
        click.secho(f"\n✅ Logged {valid_entries} entries. Deleted input file: {batch_file}", fg="green")
    else:
        click.secho("⚠️ No valid entries were logged. Input file not deleted.", fg="yellow")


@cli.command()
def schedule():
    """
    Show upcoming deliverables, dates, and dependencies by project.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT deal_name, raw_note, metadata FROM journal ORDER BY timestamp")
    rows = c.fetchall()
    conn.close()

    from collections import defaultdict
    import json

    schedule_by_project = defaultdict(list)

    for deal_name, raw_note, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            continue

        deliverables = metadata.get("deliverables", [])
        dates = metadata.get("dates", [])
        dependencies = metadata.get("dependencies", [])

        # Align lengths and zip cleanly
        for i in range(max(len(deliverables), len(dates), len(dependencies))):
            deliverable = deliverables[i] if i < len(deliverables) else None
            date = dates[i] if i < len(dates) else "Unknown date"
            depends_on = dependencies[i] if i < len(dependencies) else None
            if deliverable:
                schedule_by_project[deal_name].append((date, deliverable, depends_on))

    if not schedule_by_project:
        click.echo("No deliverables or dates found in the database.")
        return

    click.secho("\n📅 Upcoming Deliverables by Project", fg="cyan")
    for project, items in schedule_by_project.items():
        click.secho(f"\n🔹 {project}", fg="blue")
        for date, task, dep in sorted(items):
            if dep:
                click.echo(f"  {date}: {task}  ← depends on: {dep}")
            else:
                click.echo(f"  {date}: {task}")

# @cli.command()
# def log():
#     deal_name = click.prompt("Deal Name")
#     entry_type = click.prompt(
#         "Entry Type", 
#         type=click.Choice(["Meeting","Legal","Financial","DD","Note"], case_sensitive=False)
#     )
#     raw_note = click.edit(text="# Write your note above then save and close to continue.\n")
#     if raw_note is None:
#         click.echo("No note provided. Aborting.")
#         return
#     tags = click.prompt("Tags (comma-separated, no hashtags)", default="")
#     click.echo("Parsing note with AI…")
#     metadata = parse_and_distribute(raw_note.strip())
#     store_journal_entry(deal_name, entry_type, raw_note.strip(), tags, metadata)
#     click.secho("Entry logged.", fg="green")

@cli.command()
@click.argument('deal_name', required=False)
def summarize(deal_name):
    if not deal_name:
        deal_name = click.prompt("Deal Name to summarize")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, summary_text FROM summaries WHERE deal_name=? ORDER BY timestamp DESC LIMIT 1',(deal_name,))
    prev = c.fetchone()
    if prev:
        c.execute('SELECT raw_note, timestamp FROM journal WHERE deal_name=? AND timestamp>? ORDER BY timestamp',(deal_name,prev[1],))
    else:
        c.execute('SELECT raw_note, timestamp FROM journal WHERE deal_name=? ORDER BY timestamp',(deal_name,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        click.echo(f"No Journal entries for '{deal_name}'.")
        return
    content = (f"Previous summary:\n{prev[2]}\n---\nNew entries:" if prev else "Journal entries:") + "\n" + "\n".join(r[0] for r in rows)
    system_prompt = (
        f"You are an expert summarizer. Generate an updated summary for '{deal_name}' "
        "incorporating previous summary if any and these entries."
    )
    messages = [{"role":"system","content":system_prompt},{"role":"user","content":content}]
    response = call_ai_with_retry(SUMMARY_MODEL,messages)
    if not response: return
    summary = response.choices[0].message.content.strip()
    store_deal_summary(deal_name, summary, {"length":len(summary),"has_previous":bool(prev)})
    click.secho("Deal summary saved and displayed below:", fg="cyan")
    click.echo(summary)

if __name__=='__main__':
    cli()
