# Decision-tree-by-python
Decision tree example by python using Adolescent Health data 

This is Decision tree example by python.
data set: tree_addhealth.csv
Classifier method: DecisionTreeClassifier from sklearn

Decision tree analysis was performed to test nonlinear relationships among a series of explanatory variables and a binary, categorical response variable. 

My response varialbe is regular smoking.
My explanatory variables are marijuana use, Alcohol use, age, gender, and grade point average.
Since python with graphviz does not provide prune function, I used only selected variable as explanatory variables.

In the case of sex is explanatory variable, there is no indication that sex would impact regular smoking.

In the case of marijuana use experience and Alcohol use experience, Adolescents with never used marijuana and never drank alcohol were less likely to have experience with smoking [gini=0.0575, value=[110,36]]. Meanwhile,  Adolescents with used marijuana and drank alcohol were most likely to have experience with smoking [gini=0.5, value=[284,288]]. Adolescents with never used marijuana but drank alcohol were some chance to have experience with smoking [gini=0.2698, value=[731,140]]. Adolescents with used marijuana and never drank alcohol were most likely to have experience with smoking [gini=0.4611, value=[55,31]]. Overall, the correlation with marijuana is much stronger with regular smoking than Alcohol experience. 
