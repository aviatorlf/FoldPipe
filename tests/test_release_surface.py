from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.3.1"

RUNNABLE_SURFACES = (
    "README.md",
    "generate_benchmark_push.py",
    "notebooks/Quickstart.ipynb",
    "notebooks/Prion_Case_Study.ipynb",
    "scripts/hf_migration/generate_artifacts.py",
    "scripts/hf_migration/migrate.ipynb",
    "scripts/kaggle_benchmark/benchmark.ipynb",
    "scripts/kaggle_molecular/generate_artifacts.py",
    "scripts/kaggle_molecular/benchmark.ipynb",
    "scripts/kaggle_molecular_final/generate_artifacts.py",
    "scripts/kaggle_molecular_final/benchmark.ipynb",
)

PYPI_RECIPES = (
    "README.md",
    "generate_benchmark_push.py",
    "notebooks/Quickstart.ipynb",
    "notebooks/Prion_Case_Study.ipynb",
    "scripts/kaggle_benchmark/benchmark.ipynb",
    "scripts/kaggle_molecular/generate_artifacts.py",
    "scripts/kaggle_molecular/benchmark.ipynb",
    "scripts/kaggle_molecular_final/generate_artifacts.py",
    "scripts/kaggle_molecular_final/benchmark.ipynb",
)


def read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_runnable_surfaces_do_not_install_foldpipe_from_source():
    forbidden = (
        "git+https://github.com/aviatorlf/FoldPipe",
        "git clone https://github.com/aviatorlf/FoldPipe",
        "pip install -e",
    )

    for relative_path in RUNNABLE_SURFACES:
        contents = read(relative_path)
        assert not any(value in contents for value in forbidden), relative_path


def test_release_recipes_pin_the_pypi_distribution():
    requirement = f"foldpipe=={RELEASE_VERSION}"

    for relative_path in PYPI_RECIPES:
        contents = read(relative_path)
        if relative_path == "scripts/kaggle_molecular_final/generate_artifacts.py":
            assert f'FOLDPIPE_VERSION = "{RELEASE_VERSION}"' in contents
        else:
            assert requirement in contents, relative_path


def test_release_metadata_versions_match():
    assert f'version="{RELEASE_VERSION}"' in read("setup.py")
    assert f"version: {RELEASE_VERSION}" in read("CITATION.cff")
