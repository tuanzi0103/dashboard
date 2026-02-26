def is_bar_category(cat: str) -> bool:
    if not isinstance(cat, str):
        return False

    cat_upper = cat.strip().upper()

    return (
        "MTO" in cat_upper
        or "MADE TO ORDER" in cat_upper
        or cat_upper in {
            "CAFE DRINKS",
            "SMOOTHIE BAR",
            "SOUPS",
            "SWEET TREATS",
            "WRAPS & SALADS",
            "BREAKFAST BOWLS",
            "CHIA BOWLS",
        }
    )
