# Model Iterations
We trained models on both of the datasets to explore the influence that difference environments had on our results.

---

### Cards Against Humanity models
These models were initially fine-tuned on the Cards Against Humanity data, before being optimised for either toxicity or positivity. The toxicity model was optimised for 500 iterations, while the positivity models were optimised for 250 iterations. We experimented with 6 gradient accumulation steps and no gradient accumulation on the positivity models.

---

### Humicroedit models
These models were initially fine-tuned on the Humicroedit data, before being optimised for toxicity for 250 iterations. The models were evaluated on both the Humicroedit data and the Cards Against Humanity data.