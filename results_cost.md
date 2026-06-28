# Results Cost Analysis

## Model Execution Summary

| Model | Cost (€) | Execution Time |
|-------|----------|-------------------------|
| GPT-5.5 | 17.22 | X |
| Claude Opus 4.8 | 6.12 | X+(n) |
| Qwen3.6-35B-A3B | Free | D |
| Gemma-4-31B-it | Free | X+(n*4) |
| Gemma-4-12B-it | Free | X+(n*2) |

Time: 
- X: is a couple of minutes to hours
- n: in hours
- D: in days
---

### Notes
- Costs are approximate and based on API pricing as of the last execution
- In open models, I'm not reporting computing cost
- Execution times include data preprocessing and inference
- All models were tested on the same dataset
