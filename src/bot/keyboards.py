"""
Telegram keyboard builders.
"""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def news_pagination_keyboard(
    current_offset: int,
    total: int,
    limit: int = 5,
) -> InlineKeyboardMarkup:
    """
    Create pagination keyboard for news list.

    Args:
        current_offset: Current offset in the list.
        total: Total number of items.
        limit: Items per page.

    Returns:
        InlineKeyboardMarkup with prev/next buttons.
    """
    buttons = []

    # Previous button
    if current_offset > 0:
        buttons.append(
            InlineKeyboardButton(
                text="< Anterior",
                callback_data=f"news_page:{current_offset - limit}",
            )
        )

    # Next button
    if current_offset + limit < total:
        buttons.append(
            InlineKeyboardButton(
                text="Siguiente >",
                callback_data=f"news_page:{current_offset + limit}",
            )
        )

    if not buttons:
        return None

    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def article_detail_keyboard(article_id: str, source_url: str = None) -> InlineKeyboardMarkup:
    """
    Create keyboard for article detail view.

    Args:
        article_id: UUID of the article.
        source_url: URL to the original article.

    Returns:
        InlineKeyboardMarkup with action buttons.
    """
    buttons = []

    # View processed content
    buttons.append([
        InlineKeyboardButton(
            text="Ver resumen",
            callback_data=f"article_detail:{article_id}",
        )
    ])

    # Link to source
    if source_url:
        buttons.append([
            InlineKeyboardButton(
                text="Ver fuente original",
                url=source_url,
            )
        ])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def categories_keyboard(categories: list[str]) -> InlineKeyboardMarkup:
    """
    Create keyboard for category selection.

    Args:
        categories: List of available categories.

    Returns:
        InlineKeyboardMarkup with category buttons.
    """
    buttons = []
    row = []

    for i, category in enumerate(categories):
        row.append(
            InlineKeyboardButton(
                text=category.capitalize(),
                callback_data=f"category:{category}",
            )
        )
        # 2 buttons per row
        if len(row) == 2:
            buttons.append(row)
            row = []

    # Add remaining buttons
    if row:
        buttons.append(row)

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def confirm_keyboard() -> InlineKeyboardMarkup:
    """Create yes/no confirmation keyboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Si", callback_data="confirm:yes"),
                InlineKeyboardButton(text="No", callback_data="confirm:no"),
            ]
        ]
    )
