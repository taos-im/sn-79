# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Exchange agent base class.

`FinanceAgent` handles both modes: handle() dispatches on the state type, and respond_exchange() chains
through respond_simulation() to respond(), so an agent written against any one of the three methods runs
unchanged on the exchange. `_sltp_kwargs` lives there too.

FinanceExchangeAgent is kept as an alias so existing agents keep importing it. An ALIAS rather than a
subclass: a subclass carrying an abstract respond_exchange shadows the chaining in FinanceAgent, so one
implementing only respond() or respond_simulation() gets the stub's None and handle() passes that to
report(), raising AttributeError on .instructions.
"""

from taos.im.agents import FinanceAgent

FinanceExchangeAgent = FinanceAgent

__all__ = ["FinanceExchangeAgent"]
