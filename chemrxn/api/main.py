"""
FastAPI entrypoint alias for ChemRxn-Intelligence.
Enables running via:
    uvicorn chemrxn.api.main:app
as well as:
    uvicorn chemrxn.api.app:app
"""

from chemrxn.api.app import app, create_app

__all__ = ["app", "create_app"]
