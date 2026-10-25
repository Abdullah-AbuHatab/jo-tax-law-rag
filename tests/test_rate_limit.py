"""Client identity behind Cloudflare/Render: which IP a request is keyed on, and what is logged."""

import logging

from app.logs import KeyValueFormatter
from app.rate_limit import ClientIpSettings, client_ip, client_key, client_network

QUESTION = "ما هي نسبة الضريبة العامة على استيراد أي سلعة أو خدمة أو توريد أي منهما؟"
CF = ClientIpSettings(header="cf-connecting-ip", trusted_proxy_hops=1)


def test_client_ip_prefers_configured_header():
    headers = {"cf-connecting-ip": "198.51.100.7", "x-forwarded-for": "6.6.6.6, 10.1.1.1"}
    assert client_ip("10.0.0.1", headers, CF) == "198.51.100.7"
    # IPv6 is canonicalised, so two spellings of one address share a bucket
    assert client_ip("10.0.0.1", {"cf-connecting-ip": " 2001:DB8:0::1 "}, CF) == "2001:db8::1"
    # without the header configured, the same request falls back to the proxy hops logic
    assert client_ip("10.0.0.1", headers, ClientIpSettings(trusted_proxy_hops=1)) == "10.1.1.1"
    assert client_ip("10.0.0.1", headers, ClientIpSettings()) == "10.0.0.1"


def test_client_ip_ignores_spoofed_xff_when_cf_header_present(make_client):
    for forged in ("1.1.1.1", "9.9.9.9, 8.8.8.8", "not-an-ip"):
        headers = {"cf-connecting-ip": "203.0.113.9", "x-forwarded-for": forged}
        assert client_ip("10.0.0.1", headers, CF) == "203.0.113.9"

    client, _, _ = make_client(rate_limit_per_minute=2, client_ip_header="cf-connecting-ip", trusted_proxy_hops=1)
    same_client = {"CF-Connecting-IP": "203.0.113.9"}
    for forged in ("1.1.1.1", "2.2.2.2"):
        first = client.post("/ask", json={"question": QUESTION}, headers={**same_client, "X-Forwarded-For": forged})
        assert first.status_code == 200
    # a fresh forged X-Forwarded-For does not buy a fresh bucket
    third = client.post("/ask", json={"question": QUESTION}, headers={**same_client, "X-Forwarded-For": "3.3.3.3"})
    assert third.status_code == 429
    # a different real client (different CF-Connecting-IP) still has its own bucket
    other = client.post("/ask", json={"question": QUESTION}, headers={"CF-Connecting-IP": "203.0.113.10"})
    assert other.status_code == 200


def test_client_ip_invalid_header_value_falls_back_to_peer():
    no_hops = ClientIpSettings(header="cf-connecting-ip")
    for bad in ("not-an-ip", "", "1.2.3.4, 5.6.7.8", "999.1.1.1", "1.2.3.4:443", "x" * 200):
        assert client_ip("10.0.0.1", {"cf-connecting-ip": bad}, no_hops) == "10.0.0.1", bad
    # an invalid header falls through to the proxy hops logic first, then to the peer
    assert client_ip("10.0.0.1", {"cf-connecting-ip": "junk", "x-forwarded-for": "6.6.6.6, 1.2.3.4"}, CF) == "1.2.3.4"
    assert client_ip("10.0.0.1", {"cf-connecting-ip": "junk", "x-forwarded-for": "6.6.6.6, junk"}, CF) == "10.0.0.1"
    assert client_ip(None, {}, no_hops) == "unknown"


def test_client_key_is_short_salted_and_stable():
    key = client_key("203.0.113.9", "salt-a")
    assert len(key) == 12 and all(ch in "0123456789abcdef" for ch in key)
    assert key == client_key("203.0.113.9", "salt-a")
    assert key != client_key("203.0.113.10", "salt-a")
    assert key != client_key("203.0.113.9", "salt-b")


def test_request_log_has_client_key_not_ip(make_client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client, _, _ = make_client(client_ip_header="cf-connecting-ip")
    headers = {"CF-Connecting-IP": "203.0.113.9", "X-Forwarded-For": "203.0.113.77"}
    assert client.get("/health", headers=headers).status_code == 200
    assert client.post("/ask", json={"question": QUESTION}, headers=headers).status_code == 200

    request_lines = [r for r in caplog.records if r.getMessage() == "request"]
    assert len(request_lines) == 2
    assert all(len(getattr(r, "client_key", "")) == 12 for r in request_lines)
    assert request_lines[0].client_key == request_lines[1].client_key  # same client, same key
    formatter = KeyValueFormatter("%(message)s")
    for record in caplog.records:
        assert "203.0.113." not in formatter.format(record)


# ---- round 2: IPv6 clients are keyed by their /64 ----------------------------------


def _ask_from(client, ip):
    return client.post("/ask", json={"question": QUESTION}, headers={"CF-Connecting-IP": ip})


def test_client_network_groups_ipv6_by_64_and_unmaps_ipv4():
    assert client_network("2001:db8:1:2::1") == "2001:db8:1:2::/64"
    assert client_network("2001:db8:1:2:ffff:ffff:ffff:ffff") == "2001:db8:1:2::/64"
    assert client_network("2001:db8:1:3::1") == "2001:db8:1:3::/64"
    assert client_network("::ffff:203.0.113.9") == "203.0.113.9"  # IPv4-mapped IPv6 is the IPv4 client
    assert client_network("::ffff:cb00:7109") == "203.0.113.9"
    assert client_network("203.0.113.9") == "203.0.113.9"  # IPv4 unchanged
    assert client_network("unknown") == "unknown"  # not an IP (no peer): passed through
    assert client_network("testclient") == "testclient"


def test_two_addresses_in_one_64_share_a_bucket(make_client):
    client, _, _ = make_client(rate_limit_per_minute=2, client_ip_header="cf-connecting-ip")
    assert _ask_from(client, "2001:db8:1:2::1").status_code == 200
    assert _ask_from(client, "2001:db8:1:2::abcd").status_code == 200
    assert _ask_from(client, "2001:db8:1:2:dead:beef:0:1").status_code == 429  # same /64, third ask
    assert _ask_from(client, "2001:db8:1:3::1").status_code == 200  # a different /64 has its own bucket
    assert _ask_from(client, "::ffff:203.0.113.9").status_code == 200
    assert _ask_from(client, "203.0.113.9").status_code == 200
    assert _ask_from(client, "203.0.113.9").status_code == 429  # the mapped form was the same IPv4 client


def test_daily_limit_is_shared_by_one_64(make_client):
    client, _, _ = make_client(daily_ask_limit_per_client=1, client_ip_header="cf-connecting-ip")
    assert _ask_from(client, "2001:db8:aa:1::1").status_code == 200
    rotated = _ask_from(client, "2001:db8:aa:1::2")
    assert rotated.status_code == 429 and rotated.json()["error"]["code"] == "DAILY_LIMIT_REACHED"
    assert _ask_from(client, "2001:db8:aa:2::1").status_code == 200
