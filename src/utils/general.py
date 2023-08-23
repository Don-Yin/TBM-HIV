"""
Print a banner to the terminal to screen width
e.g., ---- hello world ----
"""

import shutil
from rich import print


def banner(content: str, symbol="-"):
    terminal_width, _ = shutil.get_terminal_size()
    content = " " + content.strip() + " "
    content = content.center(terminal_width, symbol)
    print(content)


if __name__ == "__main__":
    banner("Hello World")
