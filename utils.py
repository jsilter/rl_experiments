import logging
import typing

Number = typing.TypeVar("Number", int, float)


def basic_configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )
