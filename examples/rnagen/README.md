# Improving cancer type classifier with synthetic data

We demonstrate the value of generator models in boosting the performance of predictive models.
In this example, we first train a conditional variational autoencoder (CVAE) to learn the distribution of RNAseq data for cancer cell line samples. We then generate synthetic samples by sampling from the latent representation along with a random type index. Adding the synthetic samples to the original training data, we hypothesize that a model trained under the otherwise same conditions would get enhanced performance. This is validated by our experiments in training a simple type classifier network and testing the classification accuracy on the holdout data not seen by the generator.

![Test accuracy comparison](test-accuracy-comparison-20-types.png)


## Example 1

Training with 10,000 synthetic samples on the 10-type classification problem. 
```
python rnagen.py --top_k_types 10 --n_samples 10000
```

```
Train a type classifier:
Epoch 1/2

Epoch 2/2


Evaluate on test data:


Train conditional autoencoder:
Epoch 1/20

Epoch 2/20

Epoch 3/20

Epoch 4/20

Epoch 5/20

Epoch 6/20

Epoch 7/20

Epoch 8/20

Epoch 9/20

Epoch 10/20

Epoch 11/20

Epoch 12/20

Epoch 13/20

Epoch 14/20

Epoch 15/20

Epoch 16/20

Epoch 17/20

Epoch 18/20

Epoch 19/20

Epoch 20/20


Generate 10000 RNAseq samples:
Done in 0.212 seconds (47255.2 samples/s).

Train a type classifier with synthetic data:
3806 + 10000 = 13806 samples
Epoch 1/2

Epoch 2/2


Evaluate again on original test data:

Test accuracy change: +11.77% (0.7679 -> 0.8583)
```

## Example 2:

Compare and plot model performance. 

```
python rnagen.py --plot
```

```
...

Plot test accuracy using models trained with and without synthetic data:
training time: before vs after
# epochs = 1: 0.4224 vs 0.6909
# epochs = 2: 0.4772 vs 0.7372
# epochs = 3: 0.6468 vs 0.7802
# epochs = 4: 0.7450 vs 0.8175
# epochs = 5: 0.7399 vs 0.8379
# epochs = 6: 0.7843 vs 0.8542
# epochs = 7: 0.8338 vs 0.8642
# epochs = 8: 0.8217 vs 0.8649
# epochs = 9: 0.8195 vs 0.8729
# epochs = 10: 0.8246 vs 0.8625
# epochs = 11: 0.8445 vs 0.8721
# epochs = 12: 0.8539 vs 0.8831
# epochs = 13: 0.8551 vs 0.8882
# epochs = 14: 0.8658 vs 0.8848
# epochs = 15: 0.8649 vs 0.8991
# epochs = 16: 0.8743 vs 0.8896
# epochs = 17: 0.8833 vs 0.8954
# epochs = 18: 0.8777 vs 0.8984
# epochs = 19: 0.8801 vs 0.8961
# epochs = 20: 0.8773 vs 0.9010
```