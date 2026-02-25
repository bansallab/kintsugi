import nox

nox.options.default_venv_backend = "uv"

python_versions = ["3.12", "3.13", "3.14"]


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    """Run tests with pytest"""
    session.run_install(
        "uv",
        "sync",
        "--locked",
        "--group=test",
        "--no-dev",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("python", "-m", "pytest", *session.posargs)
