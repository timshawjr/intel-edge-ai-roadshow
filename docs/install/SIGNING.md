# Signing: the package is unsigned, and here is the exact step that would change that

**Verdict: the package is not signed.** There is no Authenticode signature on
`dist/M6-bundle` and **no certificate was available** to create one. No signature,
certificate, thumbprint, timestamp or certification is fabricated anywhere in this
repository, and nothing in this package should be described to a customer as signed or
certified.

The manifest records the position in the artifact itself:
`dist/M6-bundle/bundle-manifest.json` carries `"signed": false` and a `signature_note`
saying the same thing. A reader can check that field without trusting this document.

## What is unsigned, exactly

The deliverable is a **reproducible offline source bundle**, not a frozen installer: the
application source, the pinned wheels, the verified models and the one clip the application
uses, a stdlib provisioning script, and a SHA-256 manifest of every file. Signing the bundle
does not turn it into an installer, and not signing it does not make it unverified — the
manifest is verified by `python provision_offline.py --verify-only`, which checks every
file's SHA-256.

There is no code-signing certificate on this machine, no private key, and no timestamp
service account. That is why the signing step below is written down rather than performed.

## The exact commands a human with a certificate would run

`signtool` ships with the **Windows SDK**. The `<file>` is the artifact you actually ship —
an installer or a launcher if you build one, or another Authenticode-supported file. A plain
ZIP is not an Authenticode-supported type; sign the artifact inside or around it, not the
ZIP itself. `<timestamp-url>` is the **RFC 3161** timestamp service URL your certificate
authority provides; the timestamp is what keeps the signature valid after the certificate
expires.

```powershell
signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 /a <file>
signtool verify /pa /all <file>
```

- `/fd SHA256` — the file digest algorithm.
- `/tr <timestamp-url>` and `/td SHA256` — an RFC 3161 timestamp and its digest algorithm.
- `/a` — choose the best available certificate automatically. If more than one certificate
  is installed, name it explicitly with `/sha1 <thumbprint>` or `/n <subject>` instead.
- `/pa /all` on verify — use the default Authenticode policy and verify every signature,
  not just the first.

A **human** performs this step with a **code-signing certificate** they hold. It is not
performed by the application, and the application does not need to run it.

## What the absence costs

- **SmartScreen shows "unknown publisher".** Windows may warn before the package runs, and
  the warning is about the publisher's reputation, not about the manifest. Signing reduces
  this; it does not guarantee the warning disappears immediately, because reputation is
  earned over time.
- **An enterprise policy may block the package outright.** WDAC, AppLocker or a SmartScreen
  policy in an organisation can refuse an unsigned artifact with no "run anyway" path. That
  is a policy decision made by the organisation, not a defect in the bundle.
- **Some customers will not accept it.** A procurement rule that requires a signed artifact
  is a real gate, and this package does not meet it.

## What is forbidden

**Weakening a security control to work around the missing signature is forbidden.** Do not:

- add a **Defender exclusion** for the bundle or its directory;
- change or disable an **ASR rule**;
- change the **execution policy** (`Set-ExecutionPolicy`) to make a script run;
- bypass or relax **AppLocker** or WDAC;
- disable SmartScreen or add a "run anyway" instruction that depends on a weakened control.

None of these is a workaround for an unsigned package. They lower the machine's defences
for every artifact on it, and an unsigned package is not a reason to do that.

## Elevation is an install step, not a runtime step

Installing a certificate into the machine store, and running `signtool` against it, may
require **elevation**. That is an **install step**: a human runs it with the privileges it
asks for, once, before the booth. The application itself must run as a **standard user**;
nothing at runtime requires elevation, and no part of this guide asks for it.

The same rule applies to CPython itself: installing the interpreter is an install step for
a human. The bundle's provisioning script installs into a user-writable `.venv` and does not
need elevation.

## After signing

1. Run `signtool verify /pa /all <file>` and keep its output as evidence.
2. Recompute the bundle's SHA-256 if the signed artifact changed the bytes; a signature is
   part of the file and changes its hash.
3. Update `bundle-manifest.json` only by rebuilding with `tools/build_bundle.py`, never by
   hand. The manifest is generated from the files that were staged.
4. Do not describe the package as certified. Signed is not certified, and this document
   does not claim either until the commands above have been run and their output kept.
