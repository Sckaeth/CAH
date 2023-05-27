# Data Variants
Several data variants are present in the files, each of which are processed versions of the original data. All of the following variants ignore prompts which require more than one punchline. Data isn't actually on the repository as it's meant to be private.

---

### Extended data
The original data compressed into only two choices per prompt, doubled in size with generated punchline choices for each prompt. This was to aid in RLHF's reward model training.
---

### No-generation data
The same as the extended data, but without the generations, to retain the two choice format.

---

### Split data
The same as the no-generation data but split into two splits of 50% each. This was to split the data for each of the separate training processes.
