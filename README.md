**Core Functionality**

- Processes and optimizes hashcat rules from multiple files/folders
- Removes redundant rules through multiple methods:
- Textual duplicates
- Functional redundancy (rules that produce identical outputs)
- Semantic similarity (Levenshtein distance)
- Statistical analysis with Pareto optimization to suggest optimal rule limits

**Smart Processing Modes**

- Mode 1: Filter by minimum occurrence count
- Mode 2: Statistical cutoff (keep top N rules)
- Mode 3: Functional minimization (remove rules with identical behavior)
- Mode 4: Inverse mode (keep rules below certain rank)
- Mode 5: Hashcat rule validation & cleanup
- Mode 6: Levenshtein distance filtering

**Advanced Features**

- Smart processing selection: Automatically chooses CPU/GPU based on dataset size
- GPU acceleration for rule counting and validation (via OpenCL)
- Memory safety with warnings at 85% RAM usage
- Recursive file discovery (max depth 3) for rule files
- Disk mode for processing huge files without exhausting RAM
- Interactive menu for step-by-step processing


https://hcrt.pages.dev/ruleminimizer.static_workflow
