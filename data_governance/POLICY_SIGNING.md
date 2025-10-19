# Policy Signing Procedure

## Overview
This document describes the lightweight signing/attestation procedure for policy files to ensure integrity and authenticity.

## Signing Process

### 1. Generate Hash
```bash
# Generate SHA256 hash of policy file
sha256sum policies/policy.json > policy_hash.txt
```

### 2. Sign Hash
```bash
# Sign the hash with GPG (if available)
gpg --armor --detach-sign policy_hash.txt
```

### 3. Store Signature
- Save signature as `policy_hash.txt.sig`
- Include in version control
- Document in policy metadata

## Verification Process

### 1. Verify Hash
```bash
# Verify file integrity
sha256sum -c policy_hash.txt
```

### 2. Verify Signature
```bash
# Verify signature (if GPG key available)
gpg --verify policy_hash.txt.sig policy_hash.txt
```

## Example Signed Hash

```
# Example policy_hash.txt
a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456  policies/policy.json
```

## Metadata Requirements

All policy files must include:
- `__version__`: Semantic version
- `__signed_hash__`: SHA256 hash
- `__signature__`: GPG signature (if available)
- `__signed_by__`: Signer identity
- `__signed_at__`: ISO 8601 timestamp

## Security Notes

- Use strong hash algorithms (SHA256+)
- Store signatures separately from policies
- Rotate signing keys regularly
- Document key management procedures
