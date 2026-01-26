"""
Bot utility functions.
"""


def escape_md(text: str) -> str:
    """
    Escape markdown special characters for Telegram MarkdownV2.

    Args:
        text: The text to escape

    Returns:
        Text with special characters escaped for MarkdownV2 format
    """
    if not text:
        return ""
    # Characters to escape for MarkdownV2
    special_chars = [
        "_", "*", "[", "]", "(", ")", "~", "`",
        ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text
