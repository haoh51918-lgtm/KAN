"""Fail-closed governance primitives for prospective experiment execution."""

from mirage_kan.governance.authority import (
    AuthorityGuard,
    AuthorityReceipt,
    AuthoritySuperseded,
)

__all__ = ["AuthorityGuard", "AuthorityReceipt", "AuthoritySuperseded"]
