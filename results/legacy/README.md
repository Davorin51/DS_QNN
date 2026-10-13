# Historical results

These CSV files were already present before the 0.2.0 refactor. They are retained
without changing their values. The ResNet-18 summary records best accuracy
0.9245 (92.45%) over 20 epochs, 11,181,642 parameters and 3,649,611,796 estimated
FLOPs. The training script used ImageNet weights and resized CIFAR-10 to 224x224.

The old script selected its best checkpoint on the official CIFAR-10 test set
(called validation in the code). These numbers are historical observations, not
results from the new disjoint validation protocol. Hardware, seed and complete
configuration were not recorded, and no checkpoint was included. The original
README's example compression table is not used as benchmark evidence.

New scientific comparisons require fresh runs with the versioned CLI. Never mix
these legacy records with 0.2.0 summaries or infer performance gains from them.
