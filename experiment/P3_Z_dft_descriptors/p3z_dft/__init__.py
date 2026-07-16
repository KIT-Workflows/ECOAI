"""P3-Z DFT descriptor package."""

__all__ = ["main"]


def main():
    """Lazy entry point so lightweight tools can import submodules without loading config."""
    from p3z_dft.pipeline import main as _pipeline_main

    return _pipeline_main()
