# Security Policy

## Supported versions

EvoHarness is currently an alpha-stage research preview. Security fixes are
provided on the latest revision of the `main` branch; older revisions are not
maintained.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting feature:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.
3. Include affected revisions, impact, reproduction steps, and any suggested
   mitigation.

If private vulnerability reporting is unavailable, contact the repository
maintainers through their public GitHub profiles and ask for a private reporting
channel. Do not include exploit details in the initial public message.

We will acknowledge a complete report when a maintainer is available, assess
its scope, and coordinate disclosure after a fix or mitigation is ready. As a
research preview, the project does not promise a fixed response-time SLA.

## Security boundary

EvoHarness can execute model-generated code. Its built-in local subprocess
guardrails provide timeouts, process-group termination, a reduced environment,
resource limits where the operating system supports them, and best-effort proxy
black-holing. They do **not** provide a strong sandbox and do not reliably block
raw network access, filesystem access outside the candidate directory, or
operating-system exploits.

For untrusted candidates:

- run evaluation inside an isolated, disposable container or virtual machine;
- deny network access at the operating-system or infrastructure layer;
- mount task definitions, hidden tests, graders, and credentials read-only or
  outside the candidate environment;
- use least-privilege credentials and explicit spending limits; and
- inspect artifacts before sharing them because transcripts and logs may contain
  prompts, source code, or provider responses.
