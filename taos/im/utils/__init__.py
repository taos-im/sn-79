# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import re
import time

# Below this a nanosecond value is a simulation clock (a run is days long, this is three years); at or
# above it the value is a wall-clock instant, which is what the exchange path stamps on every event.
EPOCH_NS_FLOOR = 10**17


def is_wall_clock_ns(timestamp) -> bool:
    """Whether a nanosecond timestamp is an epoch instant rather than a simulation duration."""
    try:
        return int(timestamp) >= EPOCH_NS_FLOOR
    except (TypeError, ValueError):
        return False


def format_timestamp(timestamp: int) -> str:
    """Render a nanosecond timestamp for a log line.

    A simulation clock reads as a duration (``1d 02:03:04.000000000``); a wall-clock instant, which is
    what exchange-mode events carry, reads as a UTC datetime with the nanoseconds kept
    (``YYYY-MM-DD 09:37:48.804743621``). Rendering an epoch instant as a duration gave ``20706d ...``.
    """
    ts = int(timestamp)
    if not is_wall_clock_ns(ts):
        return duration_from_timestamp(ts)
    seconds, nanoseconds = divmod(ts, 1_000_000_000)
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(seconds)) + f'.{nanoseconds:09d}'


def duration_from_timestamp(timestamp : int) -> str:
    """Render a nanosecond timestamp as a human-readable duration.

    Args:
        timestamp (int): Duration in nanoseconds.

    Returns:
        str: e.g. ``'1d 02:03:04'``.
    """
    seconds, nanoseconds = divmod(timestamp, 1_000_000_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return (f"{days}d " if days > 0 else "") + f"{hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09d}"

def timestamp_from_duration(duration: str) -> int:
    """Parse a human-readable duration back into nanoseconds.

    Args:
        duration (str): e.g. ``'1d 02:03:04'``.

    Returns:
        int: The duration in nanoseconds.
    """
    match = re.match(
        r'(?:(\d+)d\s+)?(\d{2}):(\d{2}):(\d{2})\.(\d{9})$', duration.strip()
    )
    if not match:
        raise ValueError(f"Invalid duration format: {duration}")

    days, hours, minutes, seconds, nanoseconds = match.groups()
    days = int(days) if days else 0
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)

    total_seconds = (((days * 24 + hours) * 60 + minutes) * 60) + seconds
    return total_seconds * 1_000_000_000 + nanoseconds
        
def normalize(lower, upper, value):
    """Scale ``value`` into [0, 1] over the given bounds.

    Args:
        lower: Value mapping to 0.
        upper: Value mapping to 1.
        value: The value to scale.
    """
    if value is None:
        return None
    norm_range = upper - lower
    if norm_range == 0:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / norm_range))