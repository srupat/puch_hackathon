import asyncio
import os
import shutil
import zipfile
import tempfile
import json
import re
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from pydantic import BaseModel, Field
import pandas as pd

# --- Load env ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
assert TOKEN, "Please set AUTH_TOKEN in your .env"
assert MY_NUMBER, "Please set MY_NUMBER in your .env"

# --- Auth ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

mcp = FastMCP("File Manager MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool Description ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

FileManagerDescription = RichToolDescription(
    description="Perform various file operations like create, read, update, delete, list, search, move, zip/unzip, preview, summarize, convert, and generate.",
    use_when="When the user wants to interact with files in natural language.",
    side_effects="Modifies files in the server's workspace directory (ephemeral storage)."
)

# --- Workspace directory ---
WORKSPACE_DIR = Path(tempfile.gettempdir()) / "mcp_files"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_within_workspace(rel_path: Path) -> Path:
    """Resolve a relative path and ensure it stays inside WORKSPACE_DIR (prevent traversal)."""
    candidate = (WORKSPACE_DIR / rel_path).resolve()
    workspace_resolved = WORKSPACE_DIR.resolve()
    if not str(candidate).startswith(str(workspace_resolved)):
        raise ValueError("Requested path is outside the workspace.")
    return candidate

def _safe_read_text(path: Path, max_chars: int | None = None) -> str:
    """Read text safely (with optional truncation)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]"
    return text

def _summarize_text(text: str, max_sentences: int = 5) -> str:
    """A small extractive summarizer: returns the first few sentences (naive)."""
    # split into sentences (naive)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences or sentences == ['']:
        return "(no textual content)"
    # if text is short, return whole text
    if len(text.split()) <= 60:
        return text.strip()
    # otherwise return first max_sentences sentences and some stats
    selected = sentences[:max_sentences]
    summary = " ".join(selected).strip()
    # add simple stats
    words = len(text.split())
    lines = len(text.splitlines())
    return f"Summary (first {len(selected)} sentences):\n{summary}\n\n📄 {lines} lines, {words} words."

@mcp.tool(description=FileManagerDescription.model_dump_json())
async def file_manager(
    action: Annotated[str, Field(description="File operation: create, read, update, delete, list, search, move, zip, unzip, preview, summarize, convert, generate")],
    filename: Annotated[str | None, Field(description="Target file or folder name (relative to workspace)")] = None,
    content: Annotated[str | None, Field(description="Content for create/update/generate operations")] = None,
    new_name: Annotated[str | None, Field(description="New name for move/rename or output format for convert")] = None,
    search_term: Annotated[str | None, Field(description="Term to search in files")] = None,
    lines: Annotated[int | None, Field(description="Number of lines to preview")] = None
) -> str:
    try:
        action = (action or "").strip().lower()

        if action == "create":
            if not filename:
                raise ValueError("filename required for create")
            path = _resolve_within_workspace(Path(filename))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if content else "", encoding="utf-8")
            return f"File '{filename}' created."

        elif action == "read":
            if not filename:
                raise ValueError("filename required for read")
            path = WORKSPACE_DIR / filename
            if not path.exists():
                return f"File '{filename}' not found."
            return path.read_text(encoding="utf-8")

        elif action == "update":
            if not filename or content is None:
                raise ValueError("filename and content required for update")
            path = _resolve_within_workspace(Path(filename))
            if not path.exists() or not path.is_file():
                return f"File '{filename}' not found."
            # append and ensure newline separation
            with path.open("a", encoding="utf-8") as f:
                if not str(content).startswith("\n"):
                    f.write("\n")
                f.write(content)
            return f"File '{filename}' updated."

        elif action == "delete":
            if not filename:
                raise ValueError("filename required for delete")
            path = _resolve_within_workspace(Path(filename))
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
            else:
                return f"'{filename}' not found."
            return f"Deleted '{filename}'."

        elif action == "list":
            files = [str(p.relative_to(WORKSPACE_DIR)) for p in WORKSPACE_DIR.rglob("*")]
            return "\n".join(files) if files else "📂 No files found."

        elif action == "search":
            if not search_term:
                raise ValueError("search_term required for search")
            matches = []
            for file_path in WORKSPACE_DIR.rglob("*"):
                if file_path.is_file():
                    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if search_term in line:
                                matches.append(f"{file_path.relative_to(WORKSPACE_DIR)}:{line_num}: {line.strip()}")
            return "\n".join(matches) if matches else f"🔍 No matches for '{search_term}'."

        elif action == "move":
            if not filename or not new_name:
                raise ValueError("filename and new_name required for move")
            src = _resolve_within_workspace(Path(filename))
            dst = _resolve_within_workspace(Path(new_name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"Moved '{filename}' to '{new_name}'."

        elif action == "zip":
            if not filename:
                raise ValueError("filename(s) required for zip (comma-separated for multiple)")
            file_list = [WORKSPACE_DIR / f.strip() for f in filename.split(",")]
            zip_path = WORKSPACE_DIR / (new_name if new_name else "archive.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in file_list:
                    if f.exists():
                        if f.is_dir():
                            # add directory recursively
                            for root, _, files in os.walk(f):
                                for fname in files:
                                    full = Path(root) / fname
                                    zf.write(full, arcname=str(full.relative_to(WORKSPACE_DIR)))
                        else:
                            zf.write(f, arcname=str(f.relative_to(WORKSPACE_DIR)))
            return f"Created zip: {zip_path.name}"

        elif action == "unzip":
            if not filename:
                raise ValueError("zip filename required for unzip")
            zip_path = _resolve_within_workspace(Path(filename))
            if not zip_path.exists():
                return f"❌ '{filename}' not found."
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(WORKSPACE_DIR)
            return f"Unzipped '{filename}'."

        elif action == "preview":
            # preview first `lines` lines (if lines is None, show first 2000 chars)
            if not filename:
                raise ValueError("filename required for preview")
            path = _resolve_within_workspace(Path(filename))
            if not path.exists() or not path.is_file():
                return f"❌ File '{filename}' not found."
            if lines is None:
                # default preview size
                return _safe_read_text(path, max_chars=2000)
            if lines <= 0:
                raise ValueError("lines must be a positive integer for preview")
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                out_lines = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    out_lines.append(line)
            return "".join(out_lines) if out_lines else "(file is empty)"

        elif action == "summarize":
            if not filename:
                raise ValueError("filename required for summarize")
            path = _resolve_within_workspace(Path(filename))
            if not path.exists() or not path.is_file():
                return f"File '{filename}' not found."

            # CSV: return pandas describe + head
            if filename.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                    desc = df.describe(include="all").to_string()
                    head = df.head(5).to_string(index=False)
                    rows, cols = df.shape
                    return f"CSV Summary: {rows} rows x {cols} cols\n\n{desc}\n\nFirst 5 rows:\n{head}"
                except Exception as e:
                    return f"Failed to summarize CSV: {e}"

            # JSON: show shape and sample
            if filename.lower().endswith(".json"):
                try:
                    df = pd.read_json(path)
                    desc = df.describe(include="all").to_string()
                    head = df.head(5).to_string(index=False)
                    rows, cols = df.shape
                    return f"JSON Summary: {rows} rows x {cols} cols\n\n{desc}\n\nFirst 5 rows:\n{head}"
                except Exception as e:
                    # fallback to raw text summary
                    text = _safe_read_text(path, max_chars=5000)
                    return _summarize_text(text, max_sentences=5)

            # else: treat as text file
            text = _safe_read_text(path, max_chars=50000)
            return _summarize_text(text, max_sentences=6)

        elif action == "convert":
            if not filename or not new_name:
                raise ValueError("filename and new_name (target format) required for convert")
            path = _resolve_within_workspace(Path(filename))
            out_path = _resolve_within_workspace(Path(new_name))
            if filename.lower().endswith(".csv") and new_name.lower().endswith(".json"):
                df = pd.read_csv(path)
                df.to_json(out_path, orient="records", lines=False)
            elif filename.lower().endswith(".json") and new_name.lower().endswith(".csv"):
                df = pd.read_json(path)
                df.to_csv(out_path, index=False)
            elif filename.lower().endswith(".txt") and new_name.lower().endswith(".md"):
                shutil.copy(path, out_path)  # Simple rename
            else:
                return "Unsupported conversion."
            return f"🔄 Converted '{filename}' to '{new_name}'."

        elif action == "generate":
            if not filename or content is None:
                raise ValueError("filename and content prompt required for generate")
            path = _resolve_within_workspace(Path(filename))
            path.parent.mkdir(parents=True, exist_ok=True)
            # Simple generation placeholder - write content as "generated"
            if filename.lower().endswith(".csv"):
                # generate a small dummy CSV unless content instructs otherwise
                df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
                df.to_csv(path, index=False)
            else:
                path.write_text(f"Generated content (prompt):\n{content}", encoding="utf-8")
            return f"Generated '{filename}'."

        else:
            return f"Unknown action '{action}'."

    except Exception as e:
        # Wrap exception for MCP
        raise McpError(ErrorData(code="INTERNAL_ERROR", message=str(e)))


# --- Run ---
async def main():
    print(f"🚀 Starting File Manager MCP server on http://0.0.0.0:8086")
    print(f"📂 Workspace directory: {WORKSPACE_DIR}")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
