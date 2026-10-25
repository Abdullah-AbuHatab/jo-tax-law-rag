"""Create or rotate the SELECT-only Postgres role the web service connects with.

    OWNER_DATABASE_URL=postgresql://owner:...@host/db?sslmode=require \
        python -I tools/create_reader_role.py [--env-file .env]

Run once with the owner URL (and again to rotate the password). In one transaction it:

* creates role ``rag_reader`` (LOGIN and nothing else), or, if it exists, gives it a
  new random password and switches off any attribute it should not have
  (SUPERUSER, CREATEDB, CREATEROLE, REPLICATION, BYPASSRLS);
* limits it to 20 connections and sets ``statement_timeout = 5s`` and
  ``idle_in_transaction_session_timeout = 10s`` for all its sessions;
* revokes everything the role holds on the database, on schema ``public`` and on
  every table and sequence in it, then grants only CONNECT on the database, USAGE
  on ``public`` and SELECT on ``law_chunks`` and ``schema_migrations`` (default
  privileges are not touched);
* checks the result (no write privilege on any table in ``public``, no CREATE on
  the schema or the database) and refuses, changing nothing, if the role is a
  member of another role or keeps an attribute only a superuser could remove;
* writes the reader URL to ``--env-file`` (default: the repository's gitignored
  ``.env``) as ``DATABASE_URL_READER=...``, and prints only "role ready". The
  password never reaches stdout, stderr or the logs here.

By default the password is sent to the server only as a SCRAM-SHA-256 verifier
computed here (the format PostgreSQL stores), so the plaintext is never part of
any SQL text and cannot end up in the server log or pg_stat_statements.

Neon does not accept pre-hashed passwords. For Neon, ``--plaintext-password``
sends the password as a quoted literal instead. It is refused unless the owner URL
authenticates the server (``channel_binding=require`` or ``sslmode=verify-full``;
``sslmode=require`` alone encrypts but does not stop a man in the middle). The
literal may then appear in server-side statement logs or statistics: if that
matters, rotate by running the tool again.

It is not a migration: creating a login role needs a password, and migrations
are plain SQL in the repository. The tables must exist: run ``python -m app.ingest``
first.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import secrets
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

import psycopg
from psycopg import sql

REPO_ROOT = Path(__file__).resolve().parent.parent
ROLE = "rag_reader"
TABLES = ("law_chunks", "schema_migrations")
ENV_KEY = "DATABASE_URL_READER"
SCRAM_ITERATIONS = 4096  # PostgreSQL's default (scram_iterations)
# (attribute that must be off, the keyword that switches it off), in pg_roles column order.
FORBIDDEN_ATTRIBUTES = (
    ("rolsuper", "NOSUPERUSER"),
    ("rolcreatedb", "NOCREATEDB"),
    ("rolcreaterole", "NOCREATEROLE"),
    ("rolreplication", "NOREPLICATION"),
    ("rolbypassrls", "NOBYPASSRLS"),
)
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
# Set on the role (ALTER ROLE ... SET), so every session gets them without startup
# options, which Neon's pooler may reject. The web pool holds at most DB_POOL_MAX_SIZE (4).
CONNECTION_LIMIT = 20
SESSION_SETTINGS = (("statement_timeout", "5s"), ("idle_in_transaction_session_timeout", "10s"))


class UnsafeRole(Exception):
    """The role cannot be made least-privilege by this tool; the message holds no secret."""


def new_password() -> str:
    # 32 random bytes (256 bits), URL-safe (ASCII), so the URL needs no escaping.
    return secrets.token_urlsafe(32)


def scram_sha256_verifier(password: str, salt: bytes | None = None, iterations: int = SCRAM_ITERATIONS) -> str:
    """The SCRAM-SHA-256 verifier PostgreSQL stores for ``password`` (RFC 5802 / RFC 7677):
    ``SCRAM-SHA-256$<iterations>:<salt>$<StoredKey>:<ServerKey>``, base64 parts.

    Only ASCII passwords are accepted, for which SASLprep is the identity (what
    ``new_password`` produces); anything else would need SASLprep first.
    """
    if not password.isascii():
        raise ValueError("only ASCII passwords are supported")
    salt = secrets.token_bytes(16) if salt is None else salt
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("ascii"), salt, iterations)
    stored_key = hashlib.sha256(hmac.new(salted, b"Client Key", hashlib.sha256).digest()).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()

    def b64(raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    return f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def reader_url(owner_url: str, password: str) -> str:
    """``owner_url`` with the reader's credentials; host, database and query are kept."""
    parts = urlsplit(owner_url)
    if parts.scheme not in ("postgres", "postgresql") or not parts.hostname:
        raise ValueError("OWNER_DATABASE_URL must be a postgresql:// URL")
    host = parts.netloc.rpartition("@")[2]
    netloc = f"{quote(ROLE, safe='')}:{quote(password, safe='')}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def plaintext_transport_is_authenticated(owner_url: str) -> bool:
    """True when libpq will refuse the connection unless the server is authenticated:
    ``channel_binding=require`` (SCRAM bound to the TLS channel) or ``sslmode=verify-full``.
    The last occurrence of a parameter wins, as in libpq."""
    params = {key: values[-1] for key, values in parse_qs(urlsplit(owner_url).query).items()}
    return params.get("channel_binding") == "require" or params.get("sslmode") == "verify-full"


def missing_tables(conn: psycopg.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT t FROM unnest(%s::text[]) AS t WHERE to_regclass('public.' || t) IS NULL", (list(TABLES),)
    ).fetchall()
    return [row[0] for row in rows]


def _set_role_and_password(conn: psycopg.Connection, password: str, plaintext_password: bool = False) -> None:
    role = sql.Identifier(ROLE)
    # Neon refuses pre-hashed passwords; then the literal goes over the owner's TLS connection.
    verifier = sql.Literal(password if plaintext_password else scram_sha256_verifier(password))
    columns = sql.SQL(", ").join(sql.Identifier(column) for column, _ in FORBIDDEN_ATTRIBUTES)
    current = conn.execute(sql.SQL("SELECT {} FROM pg_roles WHERE rolname = %s").format(columns), (ROLE,)).fetchone()
    to_switch_off: list[str] = []
    if current is None:
        keywords = sql.SQL(" ").join(sql.SQL(keyword) for _, keyword in FORBIDDEN_ATTRIBUTES)
        statement = sql.SQL("CREATE ROLE {} WITH LOGIN {} CONNECTION LIMIT {} PASSWORD {}").format(
            role, keywords, sql.Literal(CONNECTION_LIMIT), verifier
        )
    else:
        members_of = conn.execute(
            "SELECT g.rolname FROM pg_auth_members m JOIN pg_roles g ON g.oid = m.roleid"
            " JOIN pg_roles u ON u.oid = m.member WHERE u.rolname = %s ORDER BY 1",
            (ROLE,),
        ).fetchall()
        if members_of:
            names = ", ".join(row[0] for row in members_of)
            raise UnsafeRole(f"{ROLE} is a member of {names}; drop the role and run this tool again")
        # Name an attribute only when it is on: PostgreSQL 16 lets only a role that holds
        # SUPERUSER, REPLICATION or BYPASSRLS mention it, even to switch it off.
        to_switch_off = [keyword for (_, keyword), is_on in zip(FORBIDDEN_ATTRIBUTES, current, strict=True) if is_on]
        fixes = sql.SQL("").join(sql.SQL(" " + keyword) for keyword in to_switch_off)
        statement = sql.SQL("ALTER ROLE {} WITH LOGIN{} CONNECTION LIMIT {} PASSWORD {}").format(
            role, fixes, sql.Literal(CONNECTION_LIMIT), verifier
        )
    try:
        conn.execute(statement)
    except psycopg.errors.InsufficientPrivilege as exc:
        # PostgreSQL 16: altering a role needs ADMIN OPTION on it (its creator has it), and only a
        # role that holds SUPERUSER / REPLICATION / BYPASSRLS may switch those off.
        attributes = ", ".join(to_switch_off) or "none to switch off"
        raise UnsafeRole(
            f"this owner may not alter {ROLE} (it needs ADMIN OPTION on {ROLE}, given to the role that "
            f"created it; attributes to switch off: {attributes}); drop {ROLE} and run this tool again"
        ) from exc
    except psycopg.Error as exc:
        hint = "" if plaintext_password else "; Neon does not accept pre-hashed passwords: see --plaintext-password"
        raise UnsafeRole(f"the server refused the role statement (SQLSTATE {exc.sqlstate}){hint}") from exc


def _remaining_extra_privileges(conn: psycopg.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT 'table ' || c.relname || ': ' || p.priv FROM pg_class c"
        " JOIN pg_namespace n ON n.oid = c.relnamespace CROSS JOIN unnest(%s::text[]) AS p(priv)"
        " WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'f')"
        " AND has_table_privilege(%s, c.oid, p.priv)"
        " UNION ALL SELECT 'schema public: CREATE' WHERE has_schema_privilege(%s, 'public', 'CREATE')"
        " UNION ALL SELECT 'database: CREATE' WHERE has_database_privilege(%s, current_database(), 'CREATE')",
        (list(WRITE_PRIVILEGES), ROLE, ROLE, ROLE),
    ).fetchall()
    return [row[0] for row in rows]


def ensure_reader_role(conn: psycopg.Connection, password: str, plaintext_password: bool = False) -> None:
    """Create or update the role, reset its privileges to the minimum, verify; one transaction."""
    role = sql.Identifier(ROLE)
    with conn.transaction():
        _set_role_and_password(conn, password, plaintext_password)
        for name, value in SESSION_SETTINGS:
            conn.execute(sql.SQL("ALTER ROLE {} SET {} = {}").format(role, sql.SQL(name), sql.Literal(value)))
        database = sql.Identifier(conn.execute("SELECT current_database()").fetchone()[0])
        for statement in (
            sql.SQL("REVOKE ALL ON DATABASE {} FROM {}").format(database, role),
            sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {}").format(role),
            sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {}").format(role),
            sql.SQL("REVOKE ALL ON SCHEMA public FROM {}").format(role),
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role),
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role),
            sql.SQL("GRANT SELECT ON {} TO {}").format(
                sql.SQL(", ").join(sql.Identifier("public", table) for table in TABLES), role
            ),
        ):
            conn.execute(statement)
        extra = _remaining_extra_privileges(conn)
        if extra:
            # e.g. granted by another role: only that grantor (or a superuser) can revoke it
            raise UnsafeRole(f"{ROLE} still has {', '.join(extra)}; revoke it as its grantor and run again")


def upsert_env_value(path: Path, key: str, value: str) -> None:
    """Set ``key=value`` in a dotenv file, keeping every other line; atomic replace."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in lines if not line.startswith(f"{key}=")]
    kept.append(f"{key}={value}")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")  # created with mode 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(kept) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or rotate the SELECT-only rag_reader role.")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env", help="where to write the reader URL")
    parser.add_argument(
        "--plaintext-password",
        action="store_true",
        help="send the password as a literal instead of a SCRAM verifier (Neon refuses verifiers); "
        "needs channel_binding=require or sslmode=verify-full in the owner URL",
    )
    args = parser.parse_args(argv)
    env = os.environ if env is None else env

    owner_url = env.get("OWNER_DATABASE_URL", "")
    if not owner_url:
        print("OWNER_DATABASE_URL is not set", file=sys.stderr)
        return 2
    if args.plaintext_password and not plaintext_transport_is_authenticated(owner_url):
        print(
            "--plaintext-password needs an owner URL with channel_binding=require or sslmode=verify-full "
            "(sslmode=require alone does not authenticate the server)",
            file=sys.stderr,
        )
        return 2
    password = new_password()
    try:
        url = reader_url(owner_url, password)
        with psycopg.connect(owner_url, connect_timeout=10, prepare_threshold=None) as conn:
            missing = missing_tables(conn)
            if missing:
                print(f"missing tables {missing}: run python -m app.ingest first", file=sys.stderr)
                return 2
            ensure_reader_role(conn, password, plaintext_password=args.plaintext_password)
    except (ValueError, UnsafeRole) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except psycopg.Error as exc:
        # Only the class and SQLSTATE: a server message could quote the statement.
        print(f"database error: {type(exc).__name__} (SQLSTATE {exc.sqlstate})", file=sys.stderr)
        return 1
    upsert_env_value(args.env_file, ENV_KEY, url)
    print("role ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
