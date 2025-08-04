# Entropic Resistance


<img src="enropic_resistance.png" alt="Entropic Resistance" width="400px" style="border-radius: 15px"/>


**Enhancing Group Relative Policy Optimization (GRPO) with Variational Disagreement**

This repository contains the official implementation and supplementary materials for the paper:

> **Entropic Resistance**  
> *Mawaba Pascal Dao, 2025*

## Overview

The Entropic Resistance integrates Bayesian Active Learning by Disagreement (BALD) into the Group Relative Policy Optimization (GRPO) framework to enhance exploration efficiency and robustness in reinforcement fine-tuning of language models. By quantifying epistemic uncertainty through entropy-based measures, this approach actively encourages exploration of less certain, information-rich token choices.

## TRL

grpo-vdr imports my fork of Hugging Face's <a href="https://github.com/PascalPolygon/trl" target="_blank">TRL</a> with custom Bayesian Active Learning by Disagreement (BALD) bonus</li> 

## Contents

- `scripts/` - Training and evaluation scripts.

## Key Features

- Implementation of GRPO with variational disagreement-based epistemic rewards.
- Support for different epistemic modes (`none`, `per_token`, `end_of_sequence`).
- Configurable hyperparameters for epistemic bonus influence and ensemble size.
- Efficient parallel GPU computation using Monte Carlo Dropout.

## Installation

Clone this repository:
```bash
git clone https://github.com/PascalPolygon/grpo-vdr.git
cd grpo-vdr
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

Train the GRPO model with Entropic Token Surge using the provided scripts:

```bash
bash scripts/grpo_train_multinode.sh
```

Adjust experiment configurations by editing `configs/grpo_config.yaml`.

## Cite

If you find our paper and code useful, please cite:

```bibtex
@article{mdaoentropic,
  title={Entropic Resistance},
  author={pdao2015},
  year={2025},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
}
```

## Contact

For questions or suggestions, open an issue on GitHub or reach out via email:

- **Author**: pdao2015
- **Email**: pdao2015@my.fit.edu

