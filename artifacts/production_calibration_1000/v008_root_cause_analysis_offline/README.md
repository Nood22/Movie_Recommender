# V2 1,000-user offline root-cause analysis

Read-only inputs: completed production calibration v002/v008. No model calls, TMDB data, summary generation, artifact mutation, or training.

- `root_cause_report.json`: aggregate counts and diagnostics
- `per_user_root_causes.jsonl`: exact reason, evidence, operation, and spans for each of the 867 changed users
- `per_user_root_causes.csv`: compact one-row-per-user failure index
- `contradiction_cases.jsonl`: all 11 manual-sample contradictions
- `title_leak_cases.jsonl`: all four confirmed title leaks
