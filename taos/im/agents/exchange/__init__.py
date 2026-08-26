# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Exchange agent base class.

`FinanceAgent` handles both modes: handle() dispatches on the state type, and respond_exchange() chains
through respond_simulation() to respond(), so an agent written against any one of the three methods runs
unchanged on the exchange. `_sltp_kwargs` lives there too.

FinanceExchangeAgent is kept as an alias so existing agents keep importing it. It used to be a subclass
carrying an ABSTRACT respond_exchange, which shadowed the chaining in FinanceAgent: a subclass that
implemented only respond() or respond_simulation() got the stub's None, and handle() passed that to
report(), raising AttributeError on .instructions. Aliasing removes the shadow.
"""

from taos.im.agents import FinanceAgent

FinanceExchangeAgent = FinanceAgent

__all__ = ["FinanceExchangeAgent"]
