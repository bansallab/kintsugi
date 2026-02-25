@_:
    just --list

# Start REPL with package import
[group("dev")]
repl:
    uv run python -i -c "import kintsugi as kt"

# Run mypy
[group("dev")]
types *args:
    uv run -m mypy {{ args }}

# Run tests
[group("dev")]
test *args:
    uv run -m pytest {{ args }}

# Run tests via nox
[group("dev")]
nox *args:
    uv run -m nox {{ args }}

# Build project
[group("dev")]
build *args:
    uv build {{ args }}

# Show build contents based on type
[group("dev")]
show build_type:
    @if [[ {{ build_type }} == "sdist" ]]; then tar -tf ./dist/*.tar.gz; elif [[ {{ build_type }} == "whl" ]]; then unzip -l ./dist/*.whl; else echo "Invalid choice"; fi

# Delete kintsugi cache directory
[group("dev")]
[linux]
clean-cache:
    rm -rf ~/.cache/kintsugi-data

# Sync dependencies
[group("janitorial")]
install *args:
    uv sync {{ args }}

# Delete virtual environment and various cache folders
[group("janitorial")]
clean:
    rm -rf ./.venv ./.pytest_cache ./.mypy_cache ./dist ./.nox
    find . -type d -name "__pycache__" -exec rm -r {} +

# Clean and install
[group("janitorial")]
refresh: clean install
