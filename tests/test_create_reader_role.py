"""tools/create_reader_role.py without a database: the SCRAM verifier and the SQL it sends."""

import base64
import hashlib
import hmac
import re
from contextlib import nullcontext

import pytest

from tools import create_reader_role as tool

VERIFIER = re.compile(r"^SCRAM-SHA-256\$4096:([A-Za-z0-9+/=]+)\$([A-Za-z0-9+/=]+):([A-Za-z0-9+/=]+)$")


def _keys(verifier: str) -> tuple[bytes, bytes, bytes]:
    match = VERIFIER.match(verifier)
    assert match, verifier
    return tuple(base64.b64decode(part) for part in match.groups())


def test_scram_verifier_matches_the_rfc7677_example():
    # RFC 7677 section 3: user "user", password "pencil", salt W22ZaJ0SNY7soEsUEjb6gQ==, i=4096.
    salt = base64.b64decode("W22ZaJ0SNY7soEsUEjb6gQ==")
    _, stored_key, server_key = _keys(tool.scram_sha256_verifier("pencil", salt=salt))
    auth_message = (
        b"n=user,r=rOprNGfwEbeRWgbNEkqO,"
        b"r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096,"
        b"c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0"
    )
    server_signature = hmac.new(server_key, auth_message, hashlib.sha256).digest()
    assert base64.b64encode(server_signature) == b"6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4="
    client_proof = base64.b64decode("dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=")
    client_signature = hmac.new(stored_key, auth_message, hashlib.sha256).digest()
    client_key = bytes(a ^ b for a, b in zip(client_proof, client_signature, strict=True))
    assert hashlib.sha256(client_key).digest() == stored_key


def test_scram_verifier_format_and_random_salt():
    first, second = tool.scram_sha256_verifier("pw-1"), tool.scram_sha256_verifier("pw-1")
    salt, stored_key, server_key = _keys(first)
    assert len(salt) == 16 and len(stored_key) == 32 and len(server_key) == 32
    assert first != second  # a fresh salt every time
    assert "pw-1" not in first
    with pytest.raises(ValueError):
        tool.scram_sha256_verifier("pässword")  # non-ASCII would need SASLprep


class RecordingConnection:
    """Answers the tool's catalogue queries and records every statement as sent."""

    def __init__(self, existing_role: bool) -> None:
        self.existing_role = existing_role
        self.statements: list[str] = []

    def transaction(self):
        return nullcontext()

    def execute(self, query, params=None):
        text = query.as_string(None) if hasattr(query, "as_string") else query
        self.statements.append(text + ("" if params is None else f" {params!r}"))
        rows: list[tuple] = []
        if "FROM pg_roles WHERE rolname" in text and self.existing_role:
            rows = [(False, False, False, False, False)]
        elif "current_database()" in text and "has_" not in text:
            rows = [("neondb",)]
        return _Cursor(rows)


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


@pytest.mark.parametrize("existing_role", [False, True])
def test_plaintext_password_never_reaches_the_sql(existing_role):
    conn = RecordingConnection(existing_role)
    password = tool.new_password()
    tool.ensure_reader_role(conn, password)
    sent = "\n".join(conn.statements)
    assert password not in sent
    role_statements = [s for s in conn.statements if s.startswith(("CREATE ROLE", "ALTER ROLE")) and " PASSWORD " in s]
    assert len(role_statements) == 1 and "PASSWORD 'SCRAM-SHA-256$4096:" in role_statements[0]
    if existing_role:
        # nothing to switch off: PostgreSQL 16 refuses even NOSUPERUSER from a non-superuser owner
        assert role_statements[0].startswith('ALTER ROLE "rag_reader" WITH LOGIN CONNECTION LIMIT 20 PASSWORD')
    else:
        assert "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS" in role_statements[0]
    assert any(s.startswith('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "rag_reader"') for s in conn.statements)
    grants = [s for s in conn.statements if s.startswith("GRANT")]
    assert grants == [
        'GRANT CONNECT ON DATABASE "neondb" TO "rag_reader"',
        'GRANT USAGE ON SCHEMA public TO "rag_reader"',
        'GRANT SELECT ON "public"."law_chunks", "public"."schema_migrations" TO "rag_reader"',
    ]


def test_plaintext_password_option_sends_a_literal_not_a_verifier():
    # Neon refuses pre-hashed passwords, so the owner can opt in to a plaintext literal over TLS.
    conn = RecordingConnection(existing_role=False)
    tool.ensure_reader_role(conn, "plain-pw-123", plaintext_password=True)
    create = next(s for s in conn.statements if s.startswith("CREATE ROLE"))
    assert "PASSWORD 'plain-pw-123'" in create
    assert "SCRAM-SHA-256" not in create


@pytest.mark.parametrize(
    "query",
    [
        "",  # no TLS settings at all
        "?sslmode=require",  # encrypted, but the server is not authenticated (a MITM can read the literal)
        "?sslmode=verify-ca",  # checks the CA, not the host name
        "?sslmode=prefer&channel_binding=prefer",
        "?channel_binding=disable&sslmode=require",
    ],
)
def test_plaintext_password_option_refuses_an_unauthenticated_channel(tmp_path, capsys, query):
    env = {"OWNER_DATABASE_URL": f"postgresql://owner:pw@db.example/neondb{query}"}
    code = tool.main(["--env-file", str(tmp_path / ".env"), "--plaintext-password"], env=env)
    assert code == 2
    err = capsys.readouterr().err
    assert "channel_binding=require" in err and "sslmode=verify-full" in err
    assert not (tmp_path / ".env").exists()


def test_plaintext_transport_check_accepts_channel_binding_or_verify_full():
    base = "postgresql://owner:pw@ep-x-pooler.eu-central-1.aws.neon.tech/neondb"
    assert tool.plaintext_transport_is_authenticated(f"{base}?sslmode=require&channel_binding=require")
    assert tool.plaintext_transport_is_authenticated(f"{base}?channel_binding=require")
    assert tool.plaintext_transport_is_authenticated(f"{base}?sslmode=verify-full")
    assert tool.plaintext_transport_is_authenticated(f"{base}?sslrootcert=system&sslmode=verify-full")
    assert not tool.plaintext_transport_is_authenticated(f"{base}?sslmode=require")
    assert not tool.plaintext_transport_is_authenticated(f"{base}?sslmode=verify-full&sslmode=require")  # last wins
    assert not tool.plaintext_transport_is_authenticated(f"{base}?x=channel_binding=require")


@pytest.mark.parametrize("existing_role", [False, True])
def test_role_gets_a_connection_limit_and_session_timeouts(existing_role):
    conn = RecordingConnection(existing_role)
    tool.ensure_reader_role(conn, tool.new_password())
    role_statement = next(s for s in conn.statements if s.startswith(("CREATE ROLE", "ALTER ROLE")))
    assert " CONNECTION LIMIT 20 " in role_statement
    assert "ALTER ROLE \"rag_reader\" SET statement_timeout = '5s'" in conn.statements
    assert "ALTER ROLE \"rag_reader\" SET idle_in_transaction_session_timeout = '10s'" in conn.statements
