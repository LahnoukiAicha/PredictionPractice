import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

matplotlib.rcParams["figure.figsize"] = (20, 10)

# Data Pre-Processing
df1 = pd.read_csv("student-mat.csv", sep=';')

# Drop irrelevant features
df2 = df1.drop(['sex', 'age', 'address', 'famsize', 'reason', 'Dalc', 'Walc'], axis='columns')

# One hot encoding for categorical data
categorical_columns = ['school', 'Pstatus', 'Mjob', 'Fjob', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
dummies = pd.get_dummies(df2, columns=categorical_columns, drop_first=True)
df3 = dummies

# Select the most correlated features
selected_features = ['G1', 'G2']

# Remove outliers using z-score for G1 and G2
z_score = np.abs(zscore(df3[selected_features]))
threshold = 3
df3 = df3[(z_score < threshold).all(axis=1)]

def remove_not_logical_g3_values(df, tolerance=1):
    exclude_indices = []
    
    for index, row in df.iterrows():
        g1 = row['G1']
        g2 = row['G2']
        g3 = row['G3']
        
        # Calculate the logical range for G3
        logical_g3 = (g1 + g2) / 2
        lower_bound = logical_g3 - tolerance
        upper_bound = logical_g3 + tolerance
        
        # Check if G3 is within the logical range
        if not (lower_bound <= g3 <= upper_bound):
            exclude_indices.append(index)
    
    # Drop the rows with outlier G3 values
    df_cleaned = df.drop(exclude_indices)
    
    return df_cleaned

#print(df3['G3'].shape) before removing the not logical ones  : 395rows
df4=remove_not_logical_g3_values(df3,1) 

#print(df4['G3'].shape) after :289rows

# Splitting the dataset
X = df4[selected_features]
y = df4.G3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Z-score for all the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training: lasso 
from sklearn.linear_model import Lasso
lcr_model = Lasso(alpha=0.2,fit_intercept=True) #the more alpha ~ 0.1 the best it is 
lcr_model.fit(X_train, y_train)
print("score for train : " , lcr_model.score(X_train,y_train))
print("score for test : "  ,lcr_model.score(X_test,y_test))


# Prediction
y_predicted = lcr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Function to predict G3 based on G1 and G2
def predict_note(g1, g2):
    input_data = pd.DataFrame([[g1, g2]], columns=selected_features)
    scaled_input = scaler.transform(input_data)
    predicted_g3 = lcr_model.predict(scaled_input)
    return predicted_g3[0]

# Example usage
g1 = 11
g2 = 11
predicted_g3 = predict_note(g1, g2)
print(f"Predicted G3 for G1={g1} and G2={g2}: {predicted_g3}")

# Visualization

plt.scatter(y_test, y_predicted, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted G3')
plt.show()

# the error went from 5.85 to 4.21 by feature selection , than after outliers removal ==>>0.44
