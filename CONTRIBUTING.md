# Contributing

This companion project is currently a private, pre-release workspace. Do not redistribute it or open a public mirror until the author approves a license and release destination.

## Change requirements

1. Keep the model-independent test suite deterministic and standard-library-only.
2. Do not commit credentials, personal recordings, model weights, generated speech, transcripts, or private host paths.
3. Record each external model or voice artifact in a validated manifest with an immutable revision, SHA-256 digest, source URL, and license metadata.
4. Keep irreversible actions behind explicit confirmation and preserve bounded input, output, timeout, and retention behavior.
5. Update the changelog and relevant book evidence records when behavior changes.

## Verification

From a clean checkout, create an isolated environment, install the package without dependencies, run the unit tests, and run the release audit. See `docs/REPRODUCIBILITY.md` for the exact sequence.

The project cannot accept public contributions until contribution terms and a code license are approved.
