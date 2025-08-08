# AI Learning Journey 🤖 - From Zero to Research 👨🏻‍🔬

This repository is a public record of my journey to becoming an AI Research Engineer/Scientist over the next few years.
I'm starting from scratch and aiming to build strong foundations in Mathematics, Machine Learning, Deep Learning, and Large Language Models (LLMs).

### Current Book: Grokking Deep Learning – Andrew Trask

### Progress Log
```Aug 8 - 2025```
#### Chapter 1 - What is Deep Learning?

- Deep Learning is a subset of Machine Learning focused on neural networks that learn hierarchical representations of data. 
- A neural network can be trained to learn patterns from examples without being explicitly programmed.

#### Chapter 2 – Type of Learnings

**Supervised Learning** – Models learns patterns from two datasets. Example, Model can learn pattern between historical Monday stock data and Tuesday stock data. Now model can predict stock price on any Tuesday given the day before(Monday) input data.

**Unsupervised Learning** – Models learns to convert the dataset into cluster of labeled data. Example, model learns to group dataset based on some common property. In below example model learns to categorize some data as 0 and 1 labels, where 0 represents somethings cute/addorable/animals whereas 1 represents food. The number of labels can increase to any number based on the categorization of the input dataset.
```markdown
  Cat - 0
  burger - 1
  dog - 0
  donut - 1
```

**Parameterized Learning** – Have a fixed set of learnable parameters. Imagine a fixed set of knobs., lets say 4 knobs. The model can tweak the values of these knobs until the prediction is accurate.
  **Example**, consider input data consisting of team name, venue, count of audience - {CSK, Home, 130}. Here count of audience is the knob/parameter. With this input data the model can predict the victory percentage. let's say it predicts 98% but CSK loses the match. Now the model tweaks its learning by adjusting the count of audience or any other paramters if avaialble until the prediction is accurate. 
  TLDR., Model predicts by tweaking a finite set of parameters in the input data. More like a trial & error.

**Non-Parameterized Learning** – In this method there are no fixed number of parameters. The number can change as the data grows. The model will store all the obervations(parameters) as it goes through the input data and makes predictions with the avaialable observed data. **Example**, Will a car stop or go based on the color of the traffic light?
  You store every observation you have seen:
```markdown
    Red → Stop
    Green → Go
    Yellow → Slow down
```
  When a new case comes in, you look up the closest match in your stored data. If the traffic light is "Red", you check all your stored examples of "Red" and see what happened most often       (majority vote) and predict with great accuracy.

### Next Steps
- Install Juypter, NumPy and start chapter 3
- Upload Jupyter notebooks with code + explanations
- Have I got everything in frist 2 chapters? No. I have lot of questions and not that clear on types of learnings but hoping to know about it in depth later.

### Goal
By the end of this journey, I aim to:
  - Contribute to open-source AI research projects
  - Build my own neural architectures
  - Work on LLM research (e.g., Meta’s LLaMA, GPT-like models)

**Learning in public means mistakes will happen, and I will update my notes as I correct them.**

### Notes:
**Aug 8 - 2025**
  - I don’t yet have complete clarity on all the types of learning. This repo is a live record of my understanding and will evolve as I go deeper.

