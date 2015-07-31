San Francisco Crime Classification
==================================

https://www.kaggle.com/c/sf-crime/

- 1st version: basic Adaboost. 47s training time, log_loss = 3.66252
- 2nd version: fine tuned Adaboost. 525s training time, log_loss = 2.72599
- 3rd version: 2-layers Neural Network. 770s training time, log_loss = 2.51524
- 4th version: 2-layers Neural Network with Engineered features. 823s training time, log_loss = 2.47535
- 5th version: 4-layers Neural Network with Engineered features. 3006s training time, log_loss = 2.43479
- 6th version: Fine-tuned Random Forest (1024 trees, depth 16). 3560s training time, log_loss = 2.33752