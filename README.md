# DataAugmentation
Battling the unblananced dataset problem using different data augmentation methods

The network models in the project use the area under the ROC curve (AUC)[1] as a metric for assessing prediction performance. Overall accuracy is not a suitable metric as it does not take class imbalance into account. AUC on the other hand uses recall and precision, meaning it takes advantage of the confusion matrix of the model and will thus give a more suitable measurement for models working on inbalanced datasets.

- [X] Oversampling through standard duplication
- [ ] Oversampling through duplication with small noise
- [ ] Oversampling using SMOTE 
- [ ] Oversampling using mixup 



References
1. Andrew P. Bradley - "The Use of the Area Under the ROC Curve in The Evaluation of Machine Learning Algorithms" - https://linkinghub.elsevier.com/retrieve/pii/S0031320396001422
