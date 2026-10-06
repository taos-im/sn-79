# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT


def add_simulation_args(cls, parser):
    """Add simulation engine arguments to the parser. Only called when --engine simulation."""

    parser.add_argument(
        "--simulation.seeding.fundamental.symbol.coinbase",
        type=str,
        help="Coinbase spot market symbol price to be used to seed simulation price.",
        default="BTC-USD",
    )

    parser.add_argument(
        "--simulation.seeding.fundamental.symbol.binance",
        type=str,
        help="Binance spot market symbol price to be used to seed simulation price.",
        default="btcusdt",
    )

    parser.add_argument(
        "--simulation.seeding.external.symbol.coinbase",
        type=str,
        help="Coinbase futures market symbol price to be used to seed external price used in simulation.",
        default="TAO-PERP-INTX",
    )

    parser.add_argument(
        "--simulation.seeding.external.symbol.binance",
        type=str,
        help="Binance futures market symbol price to be used to seed external price used in simulation.",
        default="taousdt",
    )

    parser.add_argument(
        "--simulation.seeding.external.sampling_seconds",
        type=int,
        help="Real time period in seconds over which external trade prices are written to file.",
        default=60,
    )

    parser.add_argument(
        "--simulation.xml_config",
        type=str,
        help="Path to XML file containing simulation configuration.",
        default="../../../simulate/trading/run/config/multiasset_simulation_0.xml",
    )

    parser.add_argument(
        "--simulation.data_service_url",
        type=str,
        help="Base URL of the MVTRX Data Service, used to fetch wallet-submitted orders "
             "each simulation tick (GET /api/v1/orders/pending). "
             "Set to empty string to disable external order fetching.",
        default="http://localhost:8080",
    )
