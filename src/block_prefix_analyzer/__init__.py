"""Block-sequence prefix reuse analyzer.

V1 scope: replay chronologically ordered request records and accumulate
prefix-reuse statistics against an in-memory prefix index.

Public surface is intentionally minimal while the package is at 0.x.y:
everything under this package MAY move or be renamed until a 1.0 release.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
