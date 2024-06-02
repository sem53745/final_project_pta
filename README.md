# PTA: Final Project

This is the final project for the Project: Text Analysis course at the University of Groningen. It tries to automatically infer whether a text is written by AI or by a human, by looking at macro patterns and distributions in the text.

### Group Members
1. Joris van Bruggen (s5723752)
1. Mervyn Bolhuis (s5119103)
1. Tieme Boerema (s5410762)
1. Jasper Kleine (s5152372)
1. Sem Bartels (s5374588)

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

> Warning: This program was created and tested using Python 3.10. We believe it should work since Python 3.5 (when support for nested typehints was introduced), but we do not guarentee functionality for versions lower than 3.10.


To run the project, run the following command:

```bash
python3 main.py <path_to_text_file> -t <human_data> <machine_data>
```

The `<path_to_text_file>` argument should be the path to the text file that you want to analyze. The `-t` flag should be followed by the paths to the human and machine training files. We recommend using the included training files `group1.jsonl` and `human.jsonl`.

Example:

```bash
python3 main.py test.jsonl -t group1.jsonl human.jsonl
```

## Presentation

Link to the [project presentation](https://docs.google.com/presentation/d/1kC95nTjriGntkb6pEcW86qXSN1RPnlaJni0SSNvNnRU/edit?usp=sharing).
