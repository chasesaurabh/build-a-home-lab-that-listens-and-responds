# Security boundary

This repository is a local CLI and research reference, not an authenticated multi-user service. Run it under an operating-system account authorized to access the input and project directory. Voice and speaker labels are not identity proof.

## Components with different guarantees

The in-memory `ControlledAssistant` example checks its declared session role, expiry, allowed operation and confirmation. Its state does not survive process exit.

`LocalLab` uses a new private project directory and SQLite for pending requests, exact-digest confirmation, revocation and duplicate results. It checks allowed services/diagnostics, request expiry and content identity. It assumes one trusted operating-system user and does not implement login, remote authentication or user roles. Its diagnostics inspect its own local database and queue. It does not administer arbitrary services.

The loopback interpretation adapter rejects remote destinations, proxy use, redirects, oversized responses, duplicate JSON keys and invalid proposals. An accepted model response can still misunderstand a negated, quoted or malicious request. Code-side allowlists and separate diagnostic approval remain necessary. Evaluation overaccept counts are reported rather than hidden.

## Data and resource limits

Device capture is deliberate and bounded. Audio validation, text limits, frame limits and output limits constrain named operations. They are not an operating-system sandbox. Logical cancellation does not guarantee immediate interruption of a model kernel or physical playback.

The archive is intended for a small private collection. Dictation assumes a single writer; replacement and receipt storage are not one crash-atomic transaction. Narration does not automatically resume after a partial failure. Retention labels do not schedule deletion.

The normal speech adapters use local files after acquisition. The optional model server listens on loopback. Installation and artifact downloads use the network. Do not expose the server port to other machines without a separately designed authentication and network boundary.

Errors and saved research records may include input text, local paths or operating-system details. Keep full logs private unless reviewed. Ignored output directories are a convenience, not a guarantee that an explicit Git add cannot publish data.

## Reporting a problem

For a non-sensitive reproducible bug, open a repository issue with a minimal synthetic case and the exact release, Python version and command. Do not publish credentials, personal audio, private transcripts or unredacted paths. If a report requires sensitive information, use GitHub’s private vulnerability reporting option when available; do not put the material in a public issue.

No security-response deadline or long-term support period is promised. Check the current source and release notes before deployment.
