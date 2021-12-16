# Data preprocessing

This project uses the MIT/BIH database for the evalaution of the model, the MIT/BIH database has over 15 beat classifications and over 20 non beat classifcations.

According to the Association for Advancement of Medical Instrumentation(AAMI) heartbeat classification standard, 1998-2008;ISO EC57, the heartbeats are classified into 5 categories namely: 
1. Normal(N)
2. Supraventricular(S) ectopic
3. Ventricular(V) ectopic
4. Fusion(F)
5. Unknown(Q)

the code in this folder handles the data preprocessing to reannotate the beats as per the above categories.