# Principled Learning of Voting Rules

This repository contains the code for my MSc thesis project at the University of Amsterdam. It builds on the **Axiomatic Deep Voting Framework** described in the paper:

> _Learning How to Vote With Principles: Axiomatic Insights Into the Collective Decisions of Neural Networks_
>  
>  Levin Hornischer, Zoi Terzopoulou
>  
>  Forthcoming in the [Journal of Artificial Intelligence Research (JAIR)](https://www.jair.org/index.php/jair/index)
>  
>  Arxiv version: [https://arxiv.org/abs/2410.16170](https://arxiv.org/abs/2410.16170)

In this project, neural networks are trained to approximate voting rules and respect axioms from social choice theory.
The training process has two stages:
- Supervised phase: the models learn to follow classical voting rules such as Borda, Plurality, and Copeland.
- Unsupervised phase: the models are trained to satisfy axioms such as Condorcet consistency and Independence.
