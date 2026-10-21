"""Shared HTTP client factory.

Why this exists: on some networks IPv6 is advertised in DNS but does not actually work. Python's resolver then tries each
IPv6 address in turn and waits for the connect timeout (here ~21 s per address) before falling back to IPv4, so EVERY new
connection cost 40-60 s (curl hides this by racing both families). Binding the client's local address to 0.0.0.0 makes
httpx use IPv4 only. Turn it off with FORCE_IPV4=false on a network where IPv6 works and you want it.
Websocket clients (e.g. a Gnani realtime STT adapter) need the same treatment."""
import httpx

from .config import get_settings


def http_client(timeout: float = 120.0) -> httpx.Client:
    transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=1) if get_settings().force_ipv4 else None
    return httpx.Client(timeout=timeout, transport=transport)


def async_http_client(timeout: float = 120.0) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1) if get_settings().force_ipv4 else None
    return httpx.AsyncClient(timeout=timeout, transport=transport)
