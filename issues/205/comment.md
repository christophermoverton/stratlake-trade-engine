Hi @XANDERCORP,

Good catch on failure recovery—this is critical for production resilience. Before diving in, I'd like clarity on:

1. **Failure Persistence**: How are failed stage states persisted? (e.g., error message, stack trace, timestamp, retry count?)
2. **Retry Logic**: Should resume attempt to retry failed stages, or only move past them?
3. **Operator Clarity**: What does the CLI summary show for failed vs. retried vs. reused stages?
4. **Recovery Testing**: Integration test showing a campaign interrupted at a failed stage, then resumed—what's the expected outcome?

Please sketch a failure state model (e.g., state transitions for FAILED → RETRIED → COMPLETED or FAILED → SKIPPED) and post back. This feeds into the checkpoint work (#202).