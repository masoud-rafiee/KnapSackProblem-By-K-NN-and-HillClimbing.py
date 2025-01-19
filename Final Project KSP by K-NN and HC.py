#-  MASOUD RAFIEE
# - INTRO TO AI - Final Project - FALL 2024
# - MACHINE LEARNING: k-NN with Hill climbing for KSP
##################################################################
import numpy as np
#import pandas for dataframe/manip./analysis
import pandas as pd
#for feeding it the datasets as Excel files
from pandas import read_csv
# for k-NN regressor
from sklearn.neighbors import KNeighborsRegressor
import random

###################### LOAD DATASET ############################
# Load the train dataset
train_data= read_csv("train_knapsack_items_dataset.csv")
df_train=pd.DataFrame(train_data)
# Load the test dataset
test_data=read_csv("test_knapsack_items_dataset.csv")
df_test=pd.DataFrame(test_data)
print (df_test.head().to_string())
print (df_train.head().to_string())


# kNN Regressor implementation: train the model on the training dataset-> to predict the density value.

###########################  TRAIN  #################### the K-NN Model
#Extracting x features from training data set
X_train=df_train[['weight', 'value','volume']] #Features columns
Y_train= df_train['density_value'] #target column
k=5
#initilizeing the k-nn with k
Knn=KNeighborsRegressor(n_neighbors=k)

#training the model with train datasets
Knn.fit(X_train,Y_train)

################## TEST #################### the model
X_test = df_test[['weight', 'value','volume']]

#predicting densities values for the test dataset
Knn.predict(X_test)
df_test['Predicted Density']=Knn.predict(X_test)

# Print the new table with knn prediction
print("\n*************************************\nTest Dataset with Predicted Densities:\n*************************************")
print (df_test.head().to_string())

################ Hill Climbing Algorithm : components (helpers first) ########################

###### Generate random neighbors ##### (a helper function for hill climbing)
def generate_neighbors(current_solution, items):
    neighbors = []
    for _ in range (3):
        new_solution = current_solution.copy()
        if random.random() < 0.5 and len(new_solution) > 0:
            new_solution.remove(random.choice(new_solution))
        else:
            new_item = random.choice(items)
            if new_item not in new_solution:
                new_solution.append(new_item)
        neighbors.append(new_solution)
    return neighbors

##### Evaluate Solution ##### (A helper function for Hill Climbing)
def evaluate_solution(solution, items, knn_model):
    total_value = 0
    total_weight = 0
    predicted_density_values = []

    for index in solution:
        item_data = items.loc[index]
        total_value += item_data['value']
        total_weight += item_data['weight']
        density = knn_model.predict(pd.DataFrame([[item_data['weight'], item_data['value'], item_data['volume']]],
                                                columns=['weight', 'value', 'volume']))[0]
        predicted_density_values.append(density)

    average_density = np.mean(predicted_density_values) if predicted_density_values else 0
    return total_value, total_weight, average_density

################## MASTER BOSS : THE HILL CLIMBING KNAPSACK : ############################
def Hill_climbing_knapsack(items, capacity, knn_model):
    current_solution = random.sample(list(items.index), k=random.randint(1, len(items)))
    best_solution = current_solution
    best_value, best_weight, best_average_density = evaluate_solution(current_solution, items, knn_model)

    while True:
        neighbors = generate_neighbors(current_solution, list(items.index))
        improved = False

        for neighbor in neighbors:
            neighbor_value, neighbor_weight, neighbor_average_density = evaluate_solution(neighbor, items, knn_model)
            # for debugging
            # print(f"Evaluating Neighbor: {neighbor}, Weight: {neighbor_weight}, Value: {neighbor_value}, Density: {neighbor_average_density}")

        # Enforce constraints for neighbors
        if (neighbor_weight <= capacity and
                neighbor_average_density <= 0.5 and
                neighbor_value > best_value):
                print(f"Neighbor Accepted: {neighbor}")
                best_solution = neighbor
                best_value = neighbor_value
                best_weight = neighbor_weight
                best_average_density = neighbor_average_density
                current_solution = neighbor
                improved = True

        if not improved:
            break
    #dEBUGGING  Validate that the final solution satisfies constraints
    #if best_weight > capacity or best_average_density > 0.5:
       #raise ValueError("Final solution does not satisfy constraints.")

    return best_solution, best_value, best_weight, best_average_density

# Run Hill Climbing and print results
knapsack_capacity = 13  # Example capacity
best_solution, best_value, best_weight, best_average_density = Hill_climbing_knapsack(df_test, knapsack_capacity, Knn)
print("\n*************************************")
print("Final Solution Summary:")
print("*************************************")
print(f"Selected Items: {best_solution}")
print(f"Knapsack Capacity: {knapsack_capacity}")
print(f"Total Value Achieved: {best_value}")
print(f"Total Weight of Items: {best_weight:.3f}")
print(f"Average Density: {best_average_density:.3f}")
print("*************************************")
