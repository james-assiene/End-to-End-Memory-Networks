# End-to-End-Memory-Networks
Repository containing my implementation of the paper "End-To-End Memory Networks" [1]

# Installation

To install the required packages, run the following command:

    pip install -r requirements.txt

This project is built upon ParlAI [2]. To train a model on a single GPU, simply run the script `mono_train.sh`.  Make sure to modify the variables `output_dir` and `datapath` in the script mentioned earlier.

[1] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory
networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors,
Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates,
Inc., 2015.

[2] Alexander H. Miller, Will Feng, Dhruv Batra, Antoine Bordes, Adam Fisch, Jiasen
Lu, Devi Parikh, and Jason Weston. 2017.
Parlai: A dialog research software platform. In
Proceedings of the Conference on Empirical Methods in Natural Language Processing, EMNLP,
pages 79–84.