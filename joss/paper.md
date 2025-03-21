---
title: 'CCS-Lib: A Python package to elicit latent knowledge from LLMs'
tags:
  - python
  - machine learning
  - interpretability
  - ai alignment
  - honest AI
authors: # sorted by num of commits
  - name: Walter Laurito
    corresponding: true
    affiliation: 3
    equal-contrib: true
  - name: Nora Belrose
    affiliation: 1 
    equal-contrib: true
  - name: Alex Mallen
    affiliation: "1, 4"
  - name: Kay Kozaronek
    affiliation: 2
  - name: Fabien Roger
    affiliation: 4
  - name: Christy Koh
    affiliation: 5
  - name: James Chua
    affiliation: 1
  - name: Jonathan NG
    affiliation: 2
  - name: Alexander Wan
    affiliation: 5
  - name: Reagan Lee
    affiliation: 5
  - name: Ben W.
    affiliation: 1
  - name: Kyle O'Brien
    affiliation: "1, 6"
  - name: Augustas Macijauskas
    affiliation: 7
  - name: Eric Mungai Kinuthia
    affiliation: 1
  - name: Marius PL
    affiliation: 2
  - name: Waree Sethapun
    affiliation: 8
  - name: Kaarel Hänni
    affiliation: 2

affiliations:
 - name: EleutherAI
   index: 1
 - name: Independent
   index: 2
 - name: FZI Research Center for Information Technology
   index: 3
 - name: Redwood Research
   index: 4
 - name: UC Berkeley
   index: 5
 - name: Microsoft
   index: 6
 - name: CAML Lab, University of Cambridge
   index: 7
 - name: Princeton University
   index: 8
date: 11 08 2023
bibliography: paper.bib

---

# Summary

`ccs` is a library designed to elicit latent knowledge ([elk](`https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/`) [@christiano2021]) from language models. It includes implementations of both the original and an enhanced version of the Contrast-Consistent Search (CCS) method and an approach based on Contrastive Representation Clustering-Top Principal Component (CRC-TPC) [@burns2022], called VINC. Designed for researchers, the `ccs` library offers features like multi-GPU support, integration with Huggingface and the training of supervised probes for comparisons.

# Statement of need

The widespread adoption of language models in real-world applications presents significant challenges, particularly the potential generation of unreliable or inaccurate content [@weidinger2021ethical; @park2023ai; @evans2021truthful; @hendrycks2021unsolved]. A notable concern is that models fine-tuned on human preferences may exacerbate existing biases or lead to convincing yet misleading outputs [@perez2022].

Recent studies indicate that it's possible to extract simulated internal beliefs or 'knowledge' from language model activations [@li2022emergent; @gurnee2023language; @azaria2023internal; @bubeck2023sparks]. While supervised probing techniques can be used for this purpose [@alain2016understanding; @marks2023geometry], they rely on labels that may be compromised by human biases or limitations in human knowledge. In some cases, it's crucial to avoid human labels altogether to allow distinguishing between a model's true knowledge and its representation of human beliefs.

These considerations have led to the development of unsupervised probing methods, such as Contrast-Consistent Search (CCS) [@burns2022]. These techniques aim to extract knowledge embedded in language models without relying on ground truth labels [@zou2023representation; @burns2022]. Such approaches offer a promising direction for uncovering the latent knowledge within language models while mitigating the influence of human biases and limitations. 

Nonetheless, current unsupervised probing methods still face challenges [@farquhar2023; @levinstein2024; @laurito2024]. These issues underscore the need for tools that enable researchers to easily train, investigate, and compare probes while analyzing the internal representations of language models. In this context, one aim of our ccs library is to provide a testbed that allows researchers to experiment with existing unsupervised probing methods—and compare them with their supervised counterparts—to elicit latent knowledge (ELK [@christiano2021]) from within the activations of a language model.

See Section [Example Usage](#Example Usage: Comparing unsupervised and supervised probes) for a simple example usage of the library.

# Implementation

The `ccs` library is developed to provide both the original and an enhanced version of the Contrast-Consistent Search (CCS) method described in the paper "Discovering Latent Knowledge in Language Models Without Supervision" by @burns2022.

Our enhanced version of CCS uses the LBFGS optimizer instead of Adam, which speeds up the training process. Furthermore, it uses learnable Platt scaling parameters to avoid the problem of sign ambiguity from the original implementation.

In addition, we have implemented an approach called VINC (Variance, Invariance, Negative Covariance). VINC is an enhanced method for eliciting latent knowledge from language models. It builds upon the Contrastive Representation Clustering—Top Principal Component (CRC-TPC) [@burns2022] approach and incorporates additional principles. VINC aims to find a direction in activation space that maximizes variance while encouraging negative correlation between statement pairs and paraphrase invariance. The method uses eigendecomposition to optimize a quadratic objective that balances these criteria. VINC can be seen as an alternative to CCS, which takes less time to train. Additional changes and more recent results on VINC and its successor can be found [here](https://blog.eleuther.ai/vincs/).

Finally, we provide a method to train supervised probes using logistic regression, allowing a comparison with unsupervised methods.

`ccs` serves as a tool for researchers to investigate the truthfulness of model outputs and explore the underlying beliefs embedded within the model. The library offers:

- A clean implementation of the enhanced and original version of CCS
- Multi-GPU Support: Efficient extraction, training, and evaluation through parallel processing
- Integration with Huggingface: Easy utilization of models and datasets from a popular source
- VINC, an alternative to CCS
- Training supervised probes with logistic regression for comparisons

For collaboration, discussion, and support, the [Eleuther AI Discord's elk channel](https://discord.com/channels/729741769192767510/1070194752785489991) provides a platform for engaging with others interested in the library or related research projects.


# Example Usage: Comparing unsupervised and supervised probes

As mentioned above, one aim of this library is to provide a testbed for experimentation with unsupervised probing methods and to compare them with their supervised counterparts. We provide a simple example of how to use the library to compare the performance of unsupervised and supervised probes.

First install the package with `pip install -e .` in the root directory. 
This should install all the necessary dependencies.

To fit reporters for the HuggingFace model `model` and dataset `dataset`, run:

```bash
ccs elicit microsoft/deberta-v2-xxlarge-mnli imdb
```

This will automatically download the model and dataset, run the model and extract the relevant representations if they
aren't cached on disk, fit reporters on them, and save the reporter checkpoints to the `ccs-reporters` folder in your
home directory. It will also evaluate the reporter classification performance on a held out test set and save it to a
CSV file in the same folder. By default the VINC reporter is used.

In addition, as an upper bound, a supervised reporter is trained using logistic regression (LR) models.

Once the run is complete, the following files are generated for analysis:

  - `results/reporters/`: Folder containing the trained reporters (CCS or VINC probes) for each layer
  - `results/cfg.yaml`: Configuration used for the run
  - `results/eval.csv`: Evaluation results for the reporters
  - `results/train_eval.csv`: Evaluation results for the reporter on the training set
  - `results/lr_eval.csv`: Evaluation results for the logistic regression models
  - `results/sweeps/`: Folder containing the sweeps for the run
  - `results/plots/`: Folder containing the plots for the run
  - `results/fingerprints.yaml`: Metadata files that store unique identifiers (fingerprints) for different dataset splits

This makes it easy to compare the performance of for a given model and dataset. Now, if you want to run a sweep to compare the performance of different models and datasets, you can use the following command:

```bash
ccs sweep --models gpt2-{medium,large,xl} --datasets imdb amazon_polarity --add_pooled
```
Additional details can be found in the of the library [README](https://github.com/EleutherAI/ccs/blob/main/README.md).

# State of the field

Most available code is often tailored to demonstrate a paper's specific methods and results rather than being user-friendly for researchers [@burns2022, @marks2023geometry, @farquhar2023]. In contrast, our work is explicitly engineered to simplify the testing, comparison, and enhancement of unsupervised methods. Consequently, as mentioned above, it offers the following features:

- A clean implementation of the enhanced and original version of CCS
- Multi-GPU Support: Efficient extraction, training, and evaluation through parallel processing
- Integration with Huggingface: Easy utilization of models and datasets from a popular source
- The method VINC, an alternative to CCS
- Training supervised probes with logistic regression for comparisons by default

# Acknowledgements

We would like to thank [EleutherAI](https://www.eleuther.ai/), [SERI MATS](https://www.serimats.org/) for supporting our work and [Long-Term Future Fund (LTFF)](https://funds.effectivealtruism.org/funds/far-future).

# References