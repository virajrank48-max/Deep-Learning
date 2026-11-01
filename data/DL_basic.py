import numpy as np

# customer information
age = 30
income = 50 

# weight 
w1 = 0.3
w2 = 0.7
bias = -30

# calculate weight sum
z = (age * w1) + (income * w2) + bias
print("Weighted Sum:", z) 

# apply sigmoid
def segmoid(z):
    return 1/ (1 + np.exp(-z))

prediction = segmoid(z)

print("Prediction:", prediction)