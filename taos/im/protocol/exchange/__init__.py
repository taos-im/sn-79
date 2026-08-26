# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
The core exchange-mode protocol classes are defined here.
These are the classes which inherit from bittensor.synapse, and are the objects which are transmitted between validator and miner via dendrite query calls.

This exchange protocol surface ships as part of this package; the exchange engine/mechanism that serves it
does not, so imports of engine-side models (e.g. ExchangeConfig) are guarded below.
"""
from typing import Optional
from taos.im.protocol.simulator import *
from taos.im.protocol.models import *
try:
    from taos.im.protocol.exchange_config import ExchangeConfig
except ImportError:
    # The exchange protocol surface (this package) ships publicly; the exchange
    # engine/mechanism, which provides the ExchangeConfig model, does not.
    # Sentinel-None so report.py / engine wiring can detect absence and raise a
    # clear error instead of a TypeError from instantiating typing.Any.
    ExchangeConfig = None  # type: ignore[assignment,misc]
from taos.common.protocol import SimulationStateUpdate, EventNotification
from taos.im.protocol.exchange.events import *
from taos.im.protocol.exchange.models import Book, Account, Balance, Order, LazyBooks, LazyAccounts
from taos.im.protocol.exchange.response import ExchangeAgentResponse, ExchangeResponseBatch
from taos.im.utils.compress import compress, decompress

class ExchangeEventNotification(EventNotification):
    """
    Base class for intelligent market simulator event notifications.
    """
    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform messages from simulator to synapse.
        """
        return ExchangeEventNotification.model_construct(event=ExchangeEvent.from_json(json))

class ExchangeStateUpdate(SimulationStateUpdate):
    """The exchange-mode state synapse: books, accounts, pools and notices for one block.

    The compact wire form mirrors the simulation synapse so one agent can consume either; ``compress`` and
    ``decompress`` carry the payload through msgpack, and ``notices`` arrives keyed by integer uid.
    """
    version : int | None = None
    timestamp : int
    block : int
    config : ExchangeConfig | dict | None = None
    pools : dict[int, dict] | None = None
    accounts : dict[int, dict[int, dict]] | None = None
    books : dict[int,Book] | None = None
    notices : dict | None = None
    response: Optional[ExchangeAgentResponse] | None  = None
    compressed : str | dict | None = None
    compression_engine : str = "lz4"

    def environment_state(self) -> dict[int, Book]:
        """
        Method returning the state of the simulation environment; in the case of intelligent markets simulation, this is the orderbook state dictionary.
        """
        return {'chain' : self.pools, 'books' : self.books}

    def agent_state(self) -> dict[int, dict[int, Account]]:
        """
        Method returning the state of the simulation agents; in the case of intelligent markets simulation, this is the accounts dictionary.
        """
        return self.accounts

    @classmethod
    def parse_dict(cls, data):
        """Build an update from a decoded wire dict without re-validation.

        Args:
            data (dict): The decoded payload.

        Returns:
            ExchangeStateUpdate: The constructed update.
        """
        ret = ExchangeStateUpdate(timestamp=data['timestamp'], block=data.get('block', 0))
        object.__setattr__(ret, "pools",    data.get("pools", {}))
        object.__setattr__(ret, "books",    data.get("books", {}))
        object.__setattr__(ret, "accounts", data.get("accounts", {}))
        object.__setattr__(ret, "notices",  data.get("notices", {}))
        return ret

    def compress(self, level=-1, engine=None, compressed_books=None):
        """Compress the update's payload for transmission.

        Args:
            level (int): Compression level; -1 uses the engine default.
            engine (str | None): Compression engine override.
            compressed_books (bytes | None): Pre-compressed books to reuse, when the caller has them.

        Returns:
            ExchangeStateUpdate | None: The update with payload compressed, or None on failure.
        """
        try:
            if engine:
                self.compression_engine = engine
            if not self.compressed:
                compressed = self.model_copy()
                if not compressed_books:
                    compressed_books = compress(
                        {bookId: book.model_dump(mode='json') if isinstance(book, Book) else book
                         for bookId, book in compressed.books.items()} if compressed.books else None,
                        level, compressed.compression_engine, self.version
                    )
                payload = {
                    "pools":    compressed.pools if compressed.pools else None,
                    "accounts": {accountId: {bookId: account.model_dump(mode='json') if isinstance(account, Account) else account for bookId, account in accounts.items()} for accountId, accounts in compressed.accounts.items()} if compressed.accounts else None,
                    "notices":  {agentId: [notice if isinstance(notice, dict) else notice.model_dump(mode='json') for notice in notices] for agentId, notices in compressed.notices.items()} if compressed.notices else None,
                    "config":   compressed.config.model_dump(mode='json') if hasattr(compressed.config, 'model_dump') else compressed.config,
                    "response": compressed.response.model_dump(mode='json') if compressed.response else None,
                }
                compressed.compressed = {
                    "books":   compressed_books,
                    "payload": compress(payload, level, compressed.compression_engine, self.version),
                }
                compressed.pools    = None
                compressed.books    = None
                compressed.accounts = None
                compressed.notices  = None
                compressed.config   = None
                compressed.response = None
                return compressed
            else:
                return self
        except Exception:
            return None

    def decompress(self, lazy=False):
        """Decompress the payload back onto this update.

        Args:
            lazy (bool): When True, books and accounts materialise on first access instead of eagerly.

        Returns:
            ExchangeStateUpdate | None: This update with fields restored, or None on failure.
        """
        try:
            if not self.compressed:
                return self
            decompressed = decompress(self.compressed, self.compression_engine, self.version)
            self.compressed = None
            self.pools    = decompressed.get('pools')
            raw_books     = decompressed.get('books')
            raw_accounts  = decompressed.get('accounts')
            if lazy:
                object.__setattr__(self, "books",    LazyBooks(raw_books) if raw_books else raw_books)
                object.__setattr__(self, "accounts", LazyAccounts(raw_accounts) if raw_accounts else raw_accounts)
            else:
                # Eagerly materialise via the same model_construct-based parsers the
                # lazy path uses. A plain `self.books = raw_books` re-runs pydantic
                # validation of the compact wire dicts against the Book/Account models
                # (which need fields like `id` absent from the compact form), raising
                # ValidationError that the except below then swallowed — leaving books
                # and accounts as None. object.__setattr__ bypasses that re-validation.
                object.__setattr__(self, "books",    LazyBooks(raw_books).parse() if raw_books else raw_books)
                object.__setattr__(self, "accounts", LazyAccounts(raw_accounts).parse() if raw_accounts else raw_accounts)
            self.notices  = parse_notices(decompressed.get('notices'))
            self.config   = decompressed.get('config')
            self.response = decompressed.get('response')
            return self
        except Exception as e:
            try:
                import bittensor as bt
                bt.logging.warning(f"ExchangeStateUpdate.decompress failed: {type(e).__name__}: {e}")
            except Exception:
                pass
            return None

    def clear_inputs(self):
        """Drop the bulky input fields after use so the object is cheap to keep."""
        self.pools    = None
        self.books    = None
        self.accounts = None
        self.notices  = None
        self.config   = None
        return self