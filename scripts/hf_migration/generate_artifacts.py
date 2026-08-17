"""Generate the self-contained Kaggle migration notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "migrate.ipynb"


def main():
    migration_code = (ROOT / "scripts/kaggle_gdrive_to_hf.py").read_text(
        encoding="utf-8"
    )
    setup_code = '''%pip install -q huggingface_hub google-api-python-client google-auth-oauthlib google-auth-httplib2

import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token_path = next(Path("/kaggle/input").rglob("token.json"), None)
if token_path is None:
    raise FileNotFoundError("No Google Drive token.json was attached to this Kaggle notebook")

os.environ["GDRIVE_SECRET_PATH"] = str(token_path)
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
'''

    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Google Drive to Hugging Face migration\n\n"
            "This operational notebook embeds the pinned repository migration driver; "
            "it does not clone or install FoldPipe from source."
        ),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_code_cell(migration_code),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    nbf.write(notebook, OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
