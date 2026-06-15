# Research skills

This document provides some general guidelines and tips for executing machine learning research projects.
Most of these things can be learned from experience, but reading them and keeping them in mind can be a shortcut towards being more productive and successful.

## Ideation

- Every project should be grounded by specific research questions, goals, or problems \- even if these questions or goals are vague or change over time. Don't try to come up with solutions before deciding on a problem being tackled.  
- Think about whether the problem is worth solving. What will change if the project is successful? Why will people care? How many people will care? This will help ensure impact.  
- At the same time, make sure your work will produce a concrete and meaningful contribution to ensure novelty.  
- Projects that build concretely on prior work tend to be more tractable.  
- The nature of research is that most ideas don't work, or at least yield mediocre results. Be prepared to iterate.  
- Try to keep your research questions modular and specific. The general question can be complex, but should be able to be broken down into tractable sub-questions.

## Literature search

- Many papers are most heavily influenced by a small number of papers. Papers that cite, or papers cited by, these "highly influential" papers can be a good starting point for a literature search.  
- Keep track of how related papers justify their contributions. For example, what experiments are done to prove the points made in the work?  
- Papers often overstate their results. Keep a skeptical mind and read closely before investing too much in past work.

## Experimentation

- After you've picked a problem and a plan of attack, try to design your first experiment as a minimum viable proof-of-concept of the idea. Toy settings or scaled-down are fine at this stage.  
- When evaluating a newly proposed method, it's best to reuse and reproduce an experimental setting proposed in past published work. If you aren't able to reproduce baseline numbers from past work, this is a big warning sign. Iterate on your reimplementation until you can resolve the differences.
- On the other hand, proposing and using a new experimental setting itself can be part of the contribution as long as you can justify it.
- Modern machine learning is heavily empirical. Running experiments on diverse experimental settings (for example, multiple domains, model families, or scales) makes results more convincing.

## Implementation

- **Codebase Evaluation & Reuse:** When starting an implementation, **mandate: Do not write complex training loops from scratch if an existing, working repository can be cloned and modified.** Evaluate existing repositories by asking:
  - Does it use standard frameworks (PyTorch, Hugging Face)?
  - Does it have a `requirements.txt` or `environment.yml`?
  - Is the training script modular and easy to modify?
- **Dataset Management:** Leverage standard Hugging Face `datasets` or `torchvision` datasets rather than writing custom download/parsing scripts to save valuable time and reduce disk I/O errors.
- Use uv for managing your environment.  
- Use git for tracking changes and commit frequently.  
- Keep a changelog.  
- As much as possible, save intermediate artifacts to avoid having to re-run expensive experiments later.  
- Use an experiment tracking system if it's helpful for you; the best system to use is the one that you will actually use to help you keep rigorous track of your work.  
- As much as possible, avoid reimplementing existing methods to avoid introducing bugs. Reuse code released with papers as much as possible, even if just as a reference point.  
- Machine learning research experiments almost always face bugs in the following order: Syntax error, mismatched shapes, out-of-memory error, distributed computation failures (if relevant), and then more subtle bugs which make the results incorrect such as bad hyperparameters. This last category is the most pernicious and often the most time-consuming to debug, but these subtle bugs are very common in research because there is rarely known "correct" behavior.  
- Keep code concise and clean.  
- Make sure experiments fail early and loudly if there is an issue.  
- Concrete tests for all parts of research code can be unnecessary, but be sure to implement sanity checks and inspect inputs and outputs to surface unexpected behavior.

## Iteration

- Before running an experiment, make sure you can answer the questions "Why am I running the experiment? What do I hope to learn from the experiment?"  
- Keep track of what you know, what you want to know, and what you hope is true. Differentiating the third one is important and can be challenging.  
- Experiments often don't go as expected. Be prepared to search for explanations for unexpected results and be open to being wrong.  
- Don't get lost doing side-quests. Once you find a good, impactful target problem, retain focus on it.  
- At the same time, be prepared to pivot. Sometimes the side quest can become the main quest.  
- If possible, change one important variable at a time, though sometimes experimental factors are entangled.  
- Unexpected results are often caused by unexpected issues \- keep an open mind and don't put too much trust in tools.  
- It can be unproductive to chase small improvements \- keep the big picture in mind.  
