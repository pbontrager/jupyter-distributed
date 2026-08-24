"""Internal proxy process launched by the Jupyter Server extension."""

from .kernel_proxy import main

if __name__ == "__main__":
    main()
