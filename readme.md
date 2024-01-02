# GabyGPT: A Basic Transformer Language Model
## Overview
This repository hosts GabyGPT, a learning project focused on understanding and implementing the basic principles of the Transformer architecture in the context of a language model. The project is inspired by educational material from freeCodeCamp, which provides comprehensive resources for diving into machine learning and natural language processing.

The Transformer model implemented here is akin to the GPT (Generative Pre-trained Transformer) models, serving as an educational tool to explore the intricacies of Transformers in natural language tasks.

## Repository Structure
- `gabyGPT_train.py`: This file contains the full architecture of the Transformer model along with the training loop. It is the core of the project, where the model is defined, trained, and saved.
- `test_bot.py`: A simple script to test and interact with the trained model. It serves as a basic demonstration of how the model can be used for generating text.
- `BPETokenizer.py`: Implements a tokenizer using the Byte Pair Encoding (BPE) algorithm. This script is crucial for preprocessing text data into a format suitable for the Transformer model.


## Implementation Details
The model, built using PyTorch, features several key components of the Transformer architecture:

- **Multi-Head Self-Attention**: Allows the model to dynamically focus on different parts of the input sequence.
- **Positional Encoding**: Provides the model with information about the order of tokens in the sequence.
- **Layer Normalization and Residual Connections**: Employed in each Transformer block to help in stabilizing and speeding up the training process.
- **Feed-Forward Network**: A component of each Transformer block for processing the sequence data.
- **Training Loop**: Facilitates the training of the model on text data, enabling it to predict subsequent tokens in a sequence.
Usage
To use this project, you need Python 3 and PyTorch. Train the model using gabyGPT_train.py. After training, you can interact with the model using test_bot.py, which provides a simple interface to generate text based on the trained model.

## Acknowledgements
This project is heavily inspired by the educational content from freeCodeCamp. Their resources have been instrumental in the development of this project, offering a solid foundation in machine learning and deep learning concepts.

## License
This project is open-sourced under the MIT license.

