# Transforming Cards Against Humanity


This is a repository revolving around the application of LMs to the task of humour recognition, in the context of Cards Against Humanity games. The aim of this work is to analyse the relative improvements reinforcement learning, via proximal policy optimisation, brings to humour recognition. Other aims also include analysing offensive humour, fill-in-the-blank style games and comparing performance against other techniques. The code in the repository is heavily inspired off TRL and prior studies applying proximal policy optimisation to generative models.

### Abstract
For time immemorial, humour has remained a driving force of day-to-day com-
munication between humans; it is a phenomenon that connects people, no matter
the distance, and brings them closer to one another. In this work we show that
BERT models, tasked with humour scoring, can be optimised towards an offensive
style of humour via proximal policy optimisation. To achieve this, we utilise a
dataset consisting of 300,000 single player rounds of the fill-in-the-blank humour
game Cards Against Humanity, introduced in a previous study, to fine-tune
and optimise these models. We introduce an implementation of proximal policy
optimisation capable of optimising language models tasked with predicting win-
ning jokes in simulated rounds of the game. Furthermore, we utilise a pre-existing
RoBERTa model trained to measure the toxicity and hate present in text as a re-
ward model. As a result, we discovered that leveraging large language models
through fine-tuning outperformed previous state of the art language-based strate-
gies in this task by 1.09%, with an accuracy of 21.44%. However, when optimising
the models fine-tuned on this data towards a darker style of humour we found that
their prediction accuracy decreased as their reward increased. Despite these de-
creases, we provide evidence that models initially fine-tuned on neutral humour,
but optimised for toxic humour, improve their performance on this game post-
optimisation. In doing so, we take a significant step towards improving humour
ranking for offensive humour and other specialised contexts of humour.

---

### Iterations
| Type                                                                                            | Description                                                                         |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [BERT (Cards Against Humanity)](https://github.com/Sckaeth/CAH/tree/main/Iterations/BERT-final) | Humour ranking models fine-tuned on the more offensive Cards Against Humanity data. |
| [BERT (Humicroedit)](https://github.com/Sckaeth/CAH/tree/main/Iterations/BERT-final-micro)      | Humour ranking models fine-tuned on the neutral-toned Humicroedit data.             |
