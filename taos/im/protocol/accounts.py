# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Accounts: balances, fees, loans and the per-book account.

Split out of taos.im.protocol.models, which re-exports every name here; import from either.
"""
from collections.abc import Mapping
from pydantic import Field
from taos.common.protocol import BaseModel
from taos.im.protocol.orders import Order, OrderCurrency


class Balance(BaseModel):
    """
    Represents an account balance for a specific currency.

    Attributes:
        currency (str): String identifier for the currency (e.g., "USD", "BTC").
        total (float): Total currency balance in the account.
        free (float): Free currency balance available for order placement.
        reserved (float): Reserved currency balance tied up in resting orders.
        initial (float | None): Initial balance for the currency at the start of the simulation or session.
    """
    c : str = Field(alias="currency")
    t : float = Field(alias="total")
    f : float = Field(alias="free")
    r : float = Field(alias="reserved")
    i : float = Field(alias="initial", default=None)

    @property
    def currency(self) -> str:
        """Readable accessor for wire field ``c``; ``currency`` is its serialized alias."""
        return self.c

    @property
    def total(self) -> float:
        """Readable accessor for wire field ``t``; ``total`` is its serialized alias."""
        return self.t

    @property
    def free(self) -> float:
        """Readable accessor for wire field ``f``; ``free`` is its serialized alias."""
        return self.f

    @property
    def reserved(self) -> float:
        """Readable accessor for wire field ``r``; ``reserved`` is its serialized alias."""
        return self.r

    @property
    def initial(self) -> float:
        """Readable accessor for wire field ``i``; ``initial`` is its serialized alias."""
        return self.i

    @classmethod
    def from_json(self, currency : str, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            currency: The order's quantity currency.
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Balance.model_construct(c=currency, t=json['t'], f=json['f'], r=json['r'], i=json['i'])


class Fees(BaseModel):
    """
    Represents account fees for a specific agent and book.

    Attributes:
        volume_traded (float): Total volume traded in the aggregation period for tiered fee assignment.
        maker_fee_rate (float): The current maker fee rate for the agent.
        taker_fee_rate (float): The current taker fee rate for the agent.
    """
    v : float | None = Field(alias="volume_traded", default=None)
    m : float = Field(alias="maker_fee_rate")
    t : float = Field(alias="taker_fee_rate")

    @property
    def volume_traded(self) -> float | None:
        """Readable accessor for wire field ``v``; ``volume_traded`` is its serialized alias."""
        return self.v

    @property
    def maker_fee_rate(self) -> float:
        """Readable accessor for wire field ``m``; ``maker_fee_rate`` is its serialized alias."""
        return self.m

    @property
    def taker_fee_rate(self) -> float:
        """Readable accessor for wire field ``t``; ``taker_fee_rate`` is its serialized alias."""
        return self.t

    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Fees.model_construct(v=json['v'], m=json['m'], t=json['t'])


class Loan(BaseModel):
    """
    Represents a loan associated with an open position for the agent.

    Attributes:
        order_id (int): ID of the order associated with the loan.
        amount (float): Total loan amount.
        currency (OrderCurrency): Currency in which the loan is denominated.
        base_collateral (float): Amount of base currency collateral posted for the loan.
        quote_collateral (float): Amount of quote currency collateral posted for the loan.
    """
    i : int = Field(alias="order_id")
    a : float = Field(alias="amount")
    c : OrderCurrency = Field(alias="currency")    
    bc : float = Field(alias="base_collateral")    
    qc : float = Field(alias="quote_collateral")

    @property
    def order_id(self) -> int:
        """Readable accessor for wire field ``i``; ``order_id`` is its serialized alias."""
        return self.i

    @property
    def amount(self) -> float:
        """Readable accessor for wire field ``a``; ``amount`` is its serialized alias."""
        return self.a

    @property
    def currency(self) -> OrderCurrency:
        """Readable accessor for wire field ``c``; ``currency`` is its serialized alias."""
        return self.c

    @property
    def base_collateral(self) -> float:
        """Readable accessor for wire field ``bc``; ``base_collateral`` is its serialized alias."""
        return self.bc

    @property
    def quote_collateral(self) -> float:
        """Readable accessor for wire field ``qc``; ``quote_collateral`` is its serialized alias."""
        return self.qc

    @classmethod
    def from_json(self, json : dict):
        """
        Method to transform simulator format model to the format required by the MarketSimulationStateUpdate synapse.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return Loan.model_construct(i=json['i'], a=json['a'], c=OrderCurrency(json['c']), bc=json['bc'], qc=json['qc'])
    
    def __str__(self):
        return f"{self.amount} {self.currency.name} [COLLAT : {self.base_collateral} BASE | {self.quote_collateral} QUOTE]"


class Account(BaseModel):
    """
    Represents an agent's trading account.

    Attributes:
        agent_id (int): The agent ID which owns the account.
        book_id (int): ID of the book on which the account is able to trade.
        base_balance (Balance): Balance object for the base currency.
        quote_balance (Balance): Balance object for the quote currency.
        base_loan (float): Amount of base currency currently borrowed.
        quote_loan (float): Amount of quote currency currently borrowed.
        base_collateral (float): Amount of base currency posted as collateral.
        quote_collateral (float): Amount of quote currency posted as collateral.
        orders (list[Order]): List of the current open orders associated to the agent.
        loans (dict[int, Loan]): Mapping from order ID to Loan objects representing open loans.
        fees (Fees | None): The current fee structure for the account.
        traded_volume (float | None): Total volume traded by the account. Defaults to None.
        delegate_stakes (dict[str, float]): This account's alpha stake on THIS book, broken down by
            the delegate hotkey holding it. See the note below on why the total is not enough.
    """
    i : int = Field(alias="agent_id")
    b : int = Field(alias="book_id")
    bb : Balance = Field(alias="base_balance")
    qb : Balance = Field(alias="quote_balance")
    bl : float = Field(alias="base_loan", default=0.0)
    ql : float = Field(alias="quote_loan", default=0.0)
    bc : float = Field(alias="base_collateral", default=0.0)
    qc : float = Field(alias="quote_collateral", default=0.0)    
    o : list[Order] = Field(alias="orders", default=[])
    l : dict[int, Loan] = Field(alias="loans", default={})
    f : Fees | None = Field(alias="fees")
    v : float | None = Field(alias="traded_volume", default=None)
    # A TOTAL IS NOT A SPENDABLE AMOUNT, and until now the total was all a miner could see.
    #
    # base_balance.free is alpha summed across every delegate the account holds stake with. A pool SELL
    # is checked against ONE delegate's stake (Pool.cpp: "rejected; delegateStake ({}) less than
    # alphaSpent ({})"), so an order sized from the total can be refused INSUFFICIENT_FUNDS while the
    # account plainly holds enough.free 1.49965345, largest single
    # delegate 0.76217809, a SELL of 1.0 refused -- and the miner had no field that would have told it.
    #
    # dTAO makes the split normal: a coldkey accumulates alpha under whichever hotkeys it staked to.
    #
    # NOT A RE-SLICE OF base_balance. This is chain stake at snapshot time; base_balance is the engine's
    # own accounting including reservations and its per-batch reconcile. They track each other closely and
    # WILL diverge transiently -- sum(delegate_stakes) == base_balance.free is not an invariant and must
    # not be asserted as one.
    #
    # `locked` is deliberately omitted: the validator has it, but the SELL path checks only `stake`, and
    # exposing a field the engine ignores invites miners to reason about it.
    #
    # Empty means NOT REPORTED (an older validator, or chain state unavailable this step) -- it does not
    # mean the account holds no stake. Default-empty keeps every existing miner working unchanged.
    ds : dict[str, float] = Field(alias="delegate_stakes", default_factory=dict)

    @property
    def delegate_stakes(self) -> dict[str, float]:
        """This account's alpha on this book, per delegate hotkey. Empty means not reported."""
        return self.ds

    @property
    def sellable_alpha(self) -> float:
        """The most alpha ONE order can sell here: the largest single delegate's stake.

        Sizing from base_balance.free is what produces INSUFFICIENT_FUNDS on an account that visibly
        holds enough, because a pool SELL draws on one delegate. Returns 0.0 when the breakdown was not
        reported, which a caller must treat as "unknown", NOT as "nothing to sell" -- falling back to
        base_balance.free is the correct behaviour there, since that is the pre-existing contract.
        """
        return max(self.ds.values()) if self.ds else 0.0

    @property
    def agent_id(self) -> int:
        """Readable accessor for wire field ``i``; ``agent_id`` is its serialized alias."""
        return self.i

    @property
    def book_id(self) -> int:
        """Readable accessor for wire field ``b``; ``book_id`` is its serialized alias."""
        return self.b

    @property
    def base_balance(self) -> Balance:
        """Readable accessor for wire field ``bb``; ``base_balance`` is its serialized alias."""
        return self.bb

    @property
    def quote_balance(self) -> Balance:
        """Readable accessor for wire field ``qb``; ``quote_balance`` is its serialized alias."""
        return self.qb

    @property
    def base_loan(self) -> float:
        """Readable accessor for wire field ``bl``; ``base_loan`` is its serialized alias."""
        return self.bl

    @property
    def quote_loan(self) -> float:
        """Readable accessor for wire field ``ql``; ``quote_loan`` is its serialized alias."""
        return self.ql

    @property
    def base_collateral(self) -> float:
        """Readable accessor for wire field ``bc``; ``base_collateral`` is its serialized alias."""
        return self.bc

    @property
    def quote_collateral(self) -> float:
        """Readable accessor for wire field ``qc``; ``quote_collateral`` is its serialized alias."""
        return self.qc

    @property
    def orders(self) -> list[Order]:
        """Readable accessor for wire field ``o``; ``orders`` is its serialized alias."""
        return self.o

    @property
    def loans(self) -> dict[int, Loan]:
        """Readable accessor for wire field ``l``; ``loans`` is its serialized alias."""
        return self.l

    @property
    def fees(self) -> Fees | None:
        """Readable accessor for wire field ``f``; ``fees`` is its serialized alias."""
        return self.f

    @property
    def traded_volume(self) -> float | None:
        """Readable accessor for wire field ``v``; ``traded_volume`` is its serialized alias."""
        return self.v
    
    @property
    def own_quote(self) -> float:
        """Quote the account actually owns: total minus loan plus collateral."""
        return self.quote_balance.total - self.quote_loan + self.quote_collateral
    
    @property
    def own_base(self) -> float:
        """Base the account actually owns: total minus loan plus collateral."""
        return self.base_balance.total - self.base_loan + self.base_collateral
    
    @classmethod
    def from_json(cls, json: dict) -> "Account":
        """
        Construct an Account from simulator JSON into an Account model,
        using model_construct and manually populating nested classes.

        Args:
            json: The simulator-format payload.

        Returns:
            The model in synapse format.
        """
        return cls.model_construct(
            i=json["i"],
            b=json["b"],
            bb=Balance.model_construct(**json["bb"]),
            qb=Balance.model_construct(**json["qb"]),
            bl=json.get("bl", 0.0),
            ql=json.get("ql", 0.0),
            bc=json.get("bc", 0.0),
            qc=json.get("qc", 0.0),
            o=[Order.from_json(o) for o in json.get("o", [])],
            l={int(k): Loan.from_json(v) for k, v in json.get("l", {}).items()},
            f=Fees.model_construct(**json["f"]) if json.get("f") else None,
            v=json.get("v"),
        )
