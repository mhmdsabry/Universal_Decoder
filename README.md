# Universal_Decoder
This is the Transformers Decoder version, based on the "Universal Transformers (Encoder-Decoder)" paper, and trained on natural data rather than algorithmic tasks.

## Introduction:

Universal Transformers, introduced by [Dehghani et al.](https://arxiv.org/abs/1807.03819) in 2018, are inspired by algorithmic extrapolation and the learning of scalable data processes and length generalisation. Universal Transformers represent a simple implementation in this direction by adding another layer of parameter sharing, besides sharing transformer layers across tokens (i.e., positions). They further share them across time, meaning all layers are reduced to one with a recurrence block. This is similar to RNNs, but here the internal connectivity is the transformer layer. Also, a main difference is that the hidden state in Universal Transformers can access/reaccess information in other layers during the recurrence, whereas in RNNs, access is only to a fixed-sized state vector.

Another important part of the paper is the discussion section, specifically where they discuss the architecture equivalent to Universal Transformers:

- Similar to the neural GPU ([Kaiser & Sutskever, 2016](https://arxiv.org/abs/1511.08228)), an architecture for learning algorithms with RNN GRU, if you remove the attention and replace the feedforward block with convolutions.
- The transformers' residual stream can be seen as memory because it acts as a refinement machine, revising token representation similar to the read and write mechanism in the Neural Turing Machine ([Graves et al., 2014](https://arxiv.org/abs/1410.5401)).

Another important aspect of Universal Transformers is adaptive computation time (ACT), which allows for dynamic computation, with time spent proportional to the difficulty of the problem—a concept that is recently gaining momentum. It is troubling that current transformers expend the same amount of computation to generate a simple word like "the" as compared to solving a math problem. This is why prompt techniques like the chain of thoughts and ReACT are most effective with difficult problems, as they allow more computational resources to be allocated to the problem (this is one of their effectiveness explanations but not the full story).

## Setup:

I have implemented a decoder-only version of Universal Transformers, which, as previously mentioned, consists essentially of a decoder layer with a recurrence. The control of how many times we should recur for each token is managed by the adaptive computation time (ACT) unit. This is the main modification from the paper. Additionally, we tested this architecture on the Tiny Shakespeare dataset, famously used by Andrej Karpathy for next character prediction. We compare this Universal Decoder against a vanilla decoder (GPT style) with 6 layers, where for the Universal Decoder the equivalent is allowing the same number for recurrence. Further details about the architecture can be found in the config file.

The experiment is straightforward: train these two models for next character prediction and monitor the differences.

## Results
### Loss:
![Loss](./assets/ut_vanilla_loss)

### Ponder Time for Next Token prediction:
"ponder time" is a term used in the paper that refers to the computation time allocated for each input. In the case of vanilla transformers, the ponder time is fixed and equals the number of layers; hence, all tokens must undergo computation across all six layers. However, for the Universal Decoder, the ponder time is flexible and ideally depends on the problem's difficulty (assuming effective learnability).
During the training of the Universal Decoder, after every 1000 iterations, I assess the model by unconditionally generating 300 tokens. I observed that more computation is allocated to generating the initial tokens, and as the generation progresses, less computation is allocated, eventually stabilising at a specific number. This behaviour is illustrated in the figure below.
![tokens_ponder_time](./assets/tokens_ponder_time)



## Run Code:
#### Environment:
```
conda env create environment.yml
conda activate ACT_env
```
#### Train UT decoder:
```
python train.py -c config.ini --context_size "${len}" --num_generated_tokens "${num_generated_tokens}" --act  --seed "${seed}" --train_batch_size "${train_batch_size}" --eval_batch_size "${eval_batch_size}" --learning_rate "${LR}" --epoch "${epoch}"
```
#### Train Vanilla decoder:
```
python train.py -c config.ini --context_size "${len}" --num_generated_tokens "${num_generated_tokens}" --seed "${seed}" --train_batch_size "${train_batch_size}" --eval_batch_size "${eval_batch_size}" --learning_rate "${LR}" --epoch "${epoch}"
```

## Future:
- Regarding the codebase, an update for length generalisation tests on the way.
- As for the analysis of results, the next step I want to take is to explore the relationship between the concept of an [attention sink](https://arxiv.org/abs/2309.17453) — where early tokens receive the most attention — and the high ponder time observed for initial tokens.

  







