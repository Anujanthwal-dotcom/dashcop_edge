"""DashCop Edge package."""

__version__ = "0.1.0"


def main():
    """Lazy loader for main CLI entrypoint."""
    from .main import main as _main
    return _main()


__all__ = ["main", "__version__"]
