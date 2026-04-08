# ANSI Colors
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_GRAY = "\033[90m"
_BLUE = "\033[94m"


def print_header(title: str):
    width = 62
    print()
    print(_CYAN + _BOLD + "=" * width + _RESET)
    print(_CYAN + _BOLD + f"  {title}".center(width) + _RESET)
    print(_CYAN + _BOLD + "=" * width + _RESET)


def print_section(title: str):
    print()
    print(_BOLD + f"── {title} " + "─" * max(0, 52 - len(title)) + _RESET)


def print_warning(msg: str):
    print(_YELLOW + f"  ⚠  {msg}" + _RESET)


def print_ok(msg: str):
    print(_GREEN + msg + _RESET)


def print_info(msg: str):
    print(_BLUE + msg + _RESET)


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


# ── Input Prompts ──────────────────────────────────────────────────────────

def prompt_float(label: str, min_val: float = None, max_val: float = None) -> float:
    while True:
        try:
            val = float(input(f"  {label}: ").strip())
            if min_val is not None and val < min_val:
                print_warning(f"Must be ≥ {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print_warning(f"Must be ≤ {max_val}.")
                continue
            return val
        except ValueError:
            print_warning("Please enter a numeric value.")


def prompt_int(label: str, min_val: int = None, max_val: int = None) -> int:
    while True:
        try:
            val = int(input(f"  {label}: ").strip())
            if min_val is not None and val < min_val:
                print_warning(f"Must be ≥ {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print_warning(f"Must be ≤ {max_val}.")
                continue
            return val
        except ValueError:
            print_warning("Please enter a whole number.")