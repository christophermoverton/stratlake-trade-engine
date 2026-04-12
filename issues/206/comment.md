Hi @XANDERCORP,

Love that you're thinking about operator control here. This is essential for real-world workflows. Before coding, please clarify:

1. **Policy Controls**: What are the specific overrides? (e.g., `--force-rerun <stage>`, `--disable-reuse`, `--invalidate-downstream <stage>`)
2. **CLI Interface**: Show example commands and expected behavior.
3. **Validation**: Can operators accidentally create invalid reuse scenarios (e.g., reuse stale output after upstream change)? How do we prevent this?
4. **Testing**: CLI integration tests showing each override in action.

Sketch the new CLI flags/config options and their semantics, then we'll align. This depends on #203 (fingerprinting) and #204 (resume logic).