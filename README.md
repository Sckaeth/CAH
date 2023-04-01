# Cards Against AI

## Description
This is a repository revolving around the application of LMs to the task of humour recognition, in the context of Cards Against Humanity games. The aim of this work is to analyse the relative improvements reinforcement learning, with human feedback, brings to humour recognition. Other aims also include analysing offensive humour, fill-in-the-blank style games and comparing performance against other techniques. All of this is subject to future change.

## Roadmap
- [X] Train GPT-2 models to experiment with text generation.
- [ ] Look into other model types.
- [ ] Split the readme files into their respective directories.

## Attempts
### GPT-2 models
These attempts aimed to train GPT-2 to produce its own punchline to the jokes presented in the game.

---

#### AT-1
The most basic of the models. Reward model had no randomness in selection when looking at "bad" punchline choices. Furthermore, the reward model was only trained off human choices.

<b>Reward Accuracy:</b> 67%-ish <br>
<b>LM Performance:</b> bad, nonsensical responses, reward model interestingly favoured offensive responses.

---

#### AT-2
The only change here was the data fed into the reward model. The data had randomised "bad" punchline choices and an extra 190,000-ish rows of generated data (from a standard GPT-2 model).

<b>Reward Accuracy:</b> 79%-ish (expected with the increase), but the lower threshold of rewards was a lot lower - providing more room for growth. <br>
<b>LM Performance:</b> even worse than before.

---

#### AT-3
I realised I needed to fine-tune the model prior to any RLHF, so I did exactly that.

<b>Reward Accuracy:</b> unchanged, should be 79%-ish as no changes were made. <br>
<b>LM Performance:</b> far better generation, tends to repeat jokes again and again in the same generation. (reward average, from 100, of -12.856)
<b> RLHF Performance:</b> TODO

---

####
