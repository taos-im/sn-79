# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Lazy views over the wire payload: levels, books and accounts materialised on access.

Split out of taos.im.protocol.models, which re-exports every name here; import from either.
"""
from collections.abc import Mapping, Sequence
from taos.common.protocol import BaseModel
from taos.im.protocol.orders import Order, Cancellation, OrderCurrency
from taos.im.protocol.book import Book, LevelInfo, TradeInfo
from taos.im.protocol.accounts import Account, Balance, Fees, Loan


class LazyLevel(Sequence):
    """
    Lazily-parsed order book level.

    This class defers construction of the `LevelInfo` and `Order` objects until their data is accessed.

    Attributes:
        _raw (dict): Raw data for the level.
        _parsed (LevelInfo | None): Parsed LevelInfo object once loaded.
    """
    __slots__ = ("_raw", "_parsed")

    def __init__(self, raw_level):
        self._raw = raw_level
        self._parsed = None

    def _load(self):
        if self._parsed is None:
            orders = [Order.model_construct(**o) for o in self._raw.get("o", [])] if self._raw.get("o") else []
            self._parsed = LevelInfo.model_construct(
                p=self._raw.get("p"),
                q=self._raw.get("q"),
                o=orders
            )
            self._raw = None

    def __getattr__(self, name):
        self._load()
        return getattr(self._parsed, name)

    def __getitem__(self, index):
        self._load()
        return self._parsed[index]

    def __len__(self):
        self._load()
        return len(self._parsed)

    def parse(self) -> LevelInfo:
        """Return fully parsed LevelInfo object."""
        self._load()
        return self._parsed


class LazyLevels(Sequence):
    """
    Collection of lazily-parsed order book levels.

    Attributes:
        _raw_levels (list[dict]): Raw level data.
        _parsed (dict[int, LazyLevel]): Cache of parsed LazyLevel objects.
    """
    def __init__(self, raw_levels):
        self._raw_levels = raw_levels
        self._parsed = {}

    def __getitem__(self, i):
        if i not in self._parsed:
            self._parsed[i] = LazyLevel(self._raw_levels[i])
        return self._parsed[i]

    def __iter__(self):
        for i in range(len(self._raw_levels)):
            yield self[i]

    def __len__(self):
        return len(self._raw_levels)

    def parse(self) -> list[LevelInfo]:
        """Parse all levels and return list of LevelInfo objects."""
        return [lvl.parse() for lvl in self]


class LazyBook(Book):
    """
    Lazily-parsed order book.

    Attributes:
        _raw (dict): Raw order book data.
        _bids (LazyLevels | None): Lazily-parsed bid levels.
        _asks (LazyLevels | None): Lazily-parsed ask levels.
        _events (list | None): Parsed events (Orders, Trades, Cancellations).
    """
    def __init__(self, raw_book):
        # Initialize pydantic v2 internal state slots that BaseModel.__init__
        # would normally set. Skipping super().__init__() (because Book's
        # required fields aren't supplied here) leaves these unset, breaking
        # any downstream access through pydantic_extra / model_dump / copy.
        # object.__setattr__ avoids BaseModel.__setattr__'s field validation.
        object.__setattr__(self, "__pydantic_fields_set__", set())
        object.__setattr__(self, "__pydantic_extra__", None)
        object.__setattr__(self, "__pydantic_private__", None)
        self._raw = raw_book
        self._bids = None
        self._asks = None
        self._events = None

    @property
    def id(self) -> int:
        """The book id, read from the raw wire dict without parsing the rest."""
        return self._raw.get("i")

    @property
    def bids(self):
        """Bid levels, parsed lazily on first access."""
        if self._bids is None:
            self._bids = LazyLevels(self._raw.get("b", []))
        return self._bids

    @property
    def asks(self):
        """Ask levels, parsed lazily on first access."""
        if self._asks is None:
            self._asks = LazyLevels(self._raw.get("a", []))
        return self._asks

    @property
    def events(self):
        """Book events, parsed lazily on first access into their event models."""
        if self._events is None:
            raw_events = self._raw.get("e", [])
            parsed_events = []
            for e in raw_events:
                ty = e.get("y")
                # model_validate (not model_construct) here: populate_by_name
                # is True on the base config, so both short and long keys are
                # accepted, AND pydantic v2 enforces required fields. Using
                # model_construct here silently produced malformed objects
                # whenever the wire dict lacked Ti/Ta/Mi/Ma for trades.
                if ty == "o":
                    parsed_events.append(Order.model_validate(e))
                elif ty == "t":
                    parsed_events.append(TradeInfo.model_validate(e))
                elif ty == "c":
                    parsed_events.append(Cancellation.model_validate(e))
                else:
                    parsed_events.append(e)
            self._events = parsed_events
        return self._events

    def parse(self) -> Book:
        """Return fully parsed Book object."""
            # r IS THE MAKER-TAKER RATIO, and it has to be passed explicitly: model_construct sets only
            # the fields named here, so an omitted one stays at its default of None however faithfully the
            # wire carried it. MTR is a property returning this field, so the accessor a miner is
            # documented to use returned None on every book in BOTH mechanisms.
        return Book.model_construct(
            i=self._raw.get("i"),
            r=self._raw.get("r"),
            b=self.bids.parse(),
            a=self.asks.parse(),
            e=self.events
        )


class LazyBooks(Mapping):
    """
    Lazily-parsed collection of order books.

    Attributes:
        _raw_books (dict[int, dict]): Raw book data keyed by book_id.
        _parsed_books (dict[int, LazyBook]): Cache of parsed LazyBook objects.
    """
    def __init__(self, raw_books: dict):
        self._raw_books = {int(k): v for k, v in raw_books.items()}
        self._parsed_books = {}

    def __getitem__(self, book_id: int):
        if book_id not in self._parsed_books:
            self._parsed_books[book_id] = LazyBook(self._raw_books[book_id])
        return self._parsed_books[book_id]

    def __iter__(self):
        return iter(self._raw_books)

    def __len__(self):
        return len(self._raw_books)

    def items(self):
        """Iterate ``(book_id, book)`` pairs, parsing each book lazily."""
        for k in self._raw_books:
            yield k, self[k]

    def values(self):
        """Iterate books, parsing each lazily."""
        for k in self._raw_books:
            yield self[k]

    def parse(self) -> dict[int, Book]:
        """Return dict of fully parsed Book objects keyed by book_id."""
        return {book_id: lb.parse() for book_id, lb in self.items()}


class LazyAccount:
    """
    Lazily-parsed trading account.

    Attributes:
        _raw (dict): Raw account data.
        _parsed (Account | None): Parsed Account object.
    """
    def __init__(self, raw_acc):
        self._raw = raw_acc
        self._parsed = None

    @property
    def data(self):
        """The parsed account, built on first access via ``model_construct`` (no re-validation)."""
        if self._parsed is None:
            bb = Balance.model_construct(**self._raw.get("bb", {}))
            qb = Balance.model_construct(**self._raw.get("qb", {}))
            orders = [Order.model_construct(**o) for o in self._raw.get("o", [])]

            loans = {}
            for k, v in self._raw.get("l", {}).items():
                loan = Loan.model_construct(**v)
                loan.c = OrderCurrency(loan.c)
                loans[int(k)] = loan

            fees = Fees.model_construct(**self._raw["f"]) if self._raw.get("f") else None

            self._parsed = Account.model_construct(
                i=self._raw.get("i"),
                b=self._raw.get("b"),
                bb=bb,
                qb=qb,
                bl=self._raw.get("bl", 0.0),
                ql=self._raw.get("ql", 0.0),
                bc=self._raw.get("bc", 0.0),
                qc=self._raw.get("qc", 0.0),
                o=orders,
                l=loans,
                f=fees,
                v=self._raw.get("v"),
                # model_construct applies defaults for absent fields, so an older validator that does not
                # send "ds" yields {} -- documented on the field as NOT REPORTED, never as "no stake".
                ds=self._raw.get("ds", {}) or {},
            )
            self._raw = None
        return self._parsed

    def __getattr__(self, name):
        return getattr(self.data, name)

    def parse(self) -> Account:
        """Return fully parsed Account object."""
        return self.data


class LazyAccounts(Mapping):
    """
    Lazily-parsed collection of agent accounts.

    Attributes:
        _raw_accounts (dict[int, dict[int, dict]]): Outer dict keyed by agent ID (uid), inner dict keyed by book_id.
        _parsed_accounts (dict[int, dict[int, LazyAccount]]): Cache of parsed LazyAccount objects.
    """
    def __init__(self, raw_accounts: dict):
        self._raw_accounts = {
            int(uid): {int(book_id): account for book_id, account in uid_accounts.items()}
            for uid, uid_accounts in raw_accounts.items()
        }
        self._parsed_accounts = {}

    def __getitem__(self, uid: int):
        if uid not in self._parsed_accounts:
            self._parsed_accounts[uid] = {
                book_id: LazyAccount(raw_acc)
                for book_id, raw_acc in self._raw_accounts[uid].items()
            }
        return self._parsed_accounts[uid]

    def __iter__(self):
        return iter(self._raw_accounts)

    def __len__(self):
        return len(self._raw_accounts)

    def items(self):
        """Iterate ``(agent_id, accounts)`` pairs, parsing lazily."""
        for k in self._raw_accounts:
            yield k, self[k]

    def values(self):
        """Iterate per-agent account maps, parsing lazily."""
        for k in self._raw_accounts:
            yield self[k]

    def parse(self) -> dict[int, dict[int, Account]]:
        """Return dict of fully parsed Account objects keyed by uid and book_id."""
        return {
            uid: {book_id: la.parse() for book_id, la in books.items()}
            for uid, books in self.items()
        }
