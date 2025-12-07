# Simple colored logging helpers using ANSI escape codes

BLUE = "\033[34m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[32m"
RESET = "\033[0m"


def log_info(msg: str) -> None:
    """Print an informational message in blue."""
    print(f"{BLUE}[INFO]{RESET} {msg}")


def log_warn(msg: str) -> None:
    """Print a warning message in yellow."""
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def log_error(msg: str) -> None:
    """Print an error message in red."""
    print(f"{RED}[ERROR]{RESET} {msg}")


def log_step(title: str) -> None:
    """Print a green step separator title."""
    print(f"\n{GREEN} {title} {RESET}")
