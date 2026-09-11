# Harbor oracle validation — 2026-09-10

All 12 trials passed, with zero Harbor trial exceptions. Backend:
`harbor_singularity_hpc.environment:SingularityWritableEnvironment`.
The agent was Harbor's oracle, using local Doocs solution files packaged as
`solution.json`; no model inference was used.

| Problem | Python | C++ | Go | Java | Python judge calls per language |
|---|---|---|---|---|---|
| 3243 | PASS | PASS | PASS | PASS | 70 |
| 3304 | PASS | PASS | PASS | PASS | 37 |
| 3366 | PASS | PASS | PASS | PASS | 258 |

3243 exercises integer and nested-array inputs with array outputs. 3304 exercises
character returns (C++ char / Go byte mapped to Python strings). 3366 exercises
HF/Doocs parameter-name differences while retaining native signatures.

Results: `jobs/leetcode-unified-oracle-validation/result.json`.
Generated task snapshots: `benchmarks/leetcode/unified_validation/`.
The run completed in approximately five minutes with concurrency two.
Toolchain logs report Python 3.12, g++ 13.3, Go 1.22.2, and Java 21.0.12.
Harbor emitted asyncio subprocess cleanup warnings after the successful result;
the process exited zero and every verifier reward was 1.

Additional checks:

- Five shared-judge regression tests passed (dynamic arguments, persistent state,
  wrong answers, worker crashes, and submission-language validation).
- Dataset validation passed for 182 problems / 728 interfaces.
- Original problem descriptions, interfaces, and canonical tests were preserved;
  only canonical HF argument names were added to dataset metadata.
- Oracle task generation succeeded for 181 problems in each language (724 tasks).
  All four variants of problem 3319 explicitly reject unsupported object transport.
- CLI generation aborts without publishing partial tasks. Opting into
  `--skip-unsupported` records exclusions in `exclusions.json`.

This is runtime and adapter validation on three problems, not a full-dataset
oracle score. Tree/object and mutation transport remain unsupported. Package
versions are not locked by these Ubuntu-based runtime recipes. See the root
README for generation, oracle, and Qwen/vLLM commands.
