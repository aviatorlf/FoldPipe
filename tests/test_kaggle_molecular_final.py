import json

from scripts.kaggle_molecular_final.generate_artifacts import build_notebook


def test_generated_manifest_cell_preserves_json_booleans():
    manifest = {
        "working_tree_dirty": True,
        "bundle_sha256": "abc123",
        "files": {},
    }
    notebook = build_notebook({}, manifest)
    assignment = next(
        line
        for line in notebook.cells[2].source.splitlines()
        if line.startswith("SOURCE_MANIFEST =")
    )
    namespace = {"json": json}

    exec(assignment, namespace)

    assert namespace["SOURCE_MANIFEST"] == manifest


def test_generated_notebook_fails_child_benchmark_errors():
    notebook = build_notebook({}, {"files": {}})

    assert "check=True" in notebook.cells[3].source
    assert 'get_secret("HF_TOKEN")' in notebook.cells[3].source
    assert 'run_environment["PYTHONPATH"]' in notebook.cells[3].source
    assert "str(Path.cwd())" in notebook.cells[3].source
