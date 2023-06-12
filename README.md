# Cards Against AI

## Description
This is a repository revolving around the application of LMs to the task of humour recognition, in the context of Cards Against Humanity games. The aim of this work is to analyse the relative improvements reinforcement learning, with human feedback, brings to humour recognition. Other aims also include analysing offensive humour, fill-in-the-blank style games and comparing performance against other techniques. All of this is subject to future change.

## Roadmap
- [X] GPT-2 models - text generation.
- [ ] BERT models - joke ranking.
- [ ] Refactor code for reward model.
- [ ] Refactor code for reinforcement learning.
- [ ] Further data cleaning, as required.
- [ ] Remove the periods from punchline jokes when training and evaluating. (preferably in the data processing section)
- [ ] Fix the evaluation's metrics towards what is needed.

---

### Iterations
| Type | Description |
| --- | --- |
| [GPT-2](https://github.com/Sckaeth/CAH/tree/main/Iterations/GPT-2) | Produces its own punchline to the jokes presented in the game. |
| [BERT](https://github.com/Sckaeth/CAH/tree/main/Iterations/BERT) | Classifies and ranks jokes formed by combinations of prompts and punchlines. |
