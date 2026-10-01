"""Virtual portfolio: the single owner of cash and position.

The portfolio is the only mutable state in the system. Its invariant is simple:
equity only changes because prices changed or because costs were charged.

Model scope: long-only, one position at a time, full entry and full exit. Partial
exits are *not* part of the model, which is what `round_trips` below assumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    initial_capital: float
    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    cash: float = field(init=False)
    units: float = 0.0
    executions: int = 0
    total_commission: float = 0.0
    round_trips: int = 0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_pct < 0 or self.slippage_pct < 0:
            raise ValueError("commission_pct and slippage_pct must be non-negative")
        self.cash = float(self.initial_capital)

    # --- helpers ---------------------------------------------------------

    def equity_at(self, price: float) -> float:
        return self.cash + self.units * price

    @property
    def in_position(self) -> bool:
        return self.units > 0

    def _fill_price(self, price: float, side: str) -> float:
        """Apply slippage against the trader: buys fill higher, sells lower."""
        if side == "buy":
            return price * (1.0 + self.slippage_pct)
        return price * (1.0 - self.slippage_pct)

    # --- mutations -------------------------------------------------------

    def buy(self, price: float, fraction: float = 1.0) -> float:
        """Buy with a fraction (0..1] of available cash. Returns units bought.

        Sizing divides the budget by the *all-in* unit cost so the commission is
        funded out of the same cash. The balance reaches exactly zero and money
        is never created.
        """
        if price <= 0:
            raise ValueError("price must be positive")
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.in_position:
            raise ValueError("already in position")

        fill = self._fill_price(price, "buy")
        budget = self.cash * fraction
        units = budget / (fill * (1.0 + self.commission_pct))
        if units <= 0:
            return 0.0

        notional = units * fill
        commission = notional * self.commission_pct
        self.cash -= notional + commission
        self.units += units
        self.total_commission += commission
        self.executions += 1
        return units

    def sell(self, price: float, fraction: float = 1.0) -> float:
        """Sell a fraction of the position. Returns cash received net of costs."""
        if price <= 0:
            raise ValueError("price must be positive")
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if not self.in_position:
            return 0.0

        units = self.units * fraction
        fill = self._fill_price(price, "sell")
        notional = units * fill
        commission = notional * self.commission_pct
        self.cash += notional - commission
        self.units -= units
        self.total_commission += commission
        self.executions += 1
        # Round trip = a position fully closed. Partial exits are outside the
        # model, so `fraction == 1.0` is the only exit that counts. If partial
        # exits are ever supported this counter must be redesigned to pair
        # entries with their closing size instead of relying on a full exit.
        if fraction == 1.0:
            self.round_trips += 1
        if self.units < 1e-12:
            self.units = 0.0
        return notional - commission