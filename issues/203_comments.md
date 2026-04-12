Hi @XANDERCORP,

Thanks for your interest. This issue focuses on deterministic input fingerprinting to decide safe stage reuse. Given the scope, here's what I'd like to see in a proposal before implementation:

**Before you start:**
1. **Fingerprint Algorithm**: What hashing strategy for inputs (e.g., SHA-256 on normalized JSON? Consider ordering sensitivity)?
2. **Input Scope**: Which fields are included—config, upstream artifact references, runtime options? Any exclusions?
3. **Fingerprint Storage**: Where/how are fingerprints persisted (in checkpoint? separate index)?
4. **Validation Plan**: Unit tests for determinism across schema changes + integration test showing safe/unsafe reuse scenarios.

This is linked to #202 (checkpoints), so align your fingerprint design with that work. Propose a sketch and we'll align before implementation.