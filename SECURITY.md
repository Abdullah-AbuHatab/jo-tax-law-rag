# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:

- Use GitHub's private reporting: the **Security** tab of this repository →
  **Report a vulnerability**
  (<https://github.com/Abdullah-AbuHatab/jo-tax-law-rag/security/advisories/new>).
- Include what you found, the steps to reproduce it, and the impact you expect.
  Do not include real secrets or other people's data in the report.

You should get a first answer within 7 days. Once a fix is deployed, the
advisory can be published with credit to you, if you want it.

## Scope

In scope:

- The code in this repository: the FastAPI service (`app/`), its UI
  (`app/static/`), the ingest and evaluation tools, the Dockerfile and the CI
  configuration.
- The public demo deployed from this repository, for problems in this code or
  its configuration (for example: bypassing the rate limits or the daily budgets,
  reading or changing the database through the API, script injection in the UI,
  or leaked secrets).

Out of scope:

- The third-party services the demo uses (Render, Neon, Cloudflare, Google
  Gemini, OpenRouter). Report those to the vendor.
- Volumetric denial of service, load testing, and automated scanning that sends
  heavy traffic to the demo. The demo runs on a free plan; please test locally
  (see "Run locally" in the README) instead.
- The risks the README lists under "Accepted risks", for example that the in-memory limits and daily budgets
  reset when the free instance restarts after it has been idle.
- Wrong or incomplete legal answers that come with valid citations. That is an
  answer-quality issue, so please open a normal issue. The service is not legal
  advice.
- Social engineering, and attacks that need physical access or an already
  compromised machine.

## No bounty

This is a personal, non-commercial project. There is no bug bounty and no
payment for reports. Good-faith research that follows this policy is welcome
and appreciated.
