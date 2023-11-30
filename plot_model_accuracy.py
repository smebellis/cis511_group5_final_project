import json
import matplotlib.pyplot as plt

# Names of the models 
model_names = ['Sentiment_Analysis_TFIDF_DT', 'SVM', 'BOW']

accuracies = []

# Read accuracy from each file
for model in model_names:
    try:
        with open(f'{model}_accuracy.json', 'r') as file:
            data = json.load(file)
            accuracies.append(data['accuracy'])
    except FileNotFoundError:
        print(f"Accuracy file for {model} not found.")
        accuracies.append(None)

# Plotting
plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies Comparison')

# Display accuracies on the graph
for i, v in enumerate(accuracies):
    if v is not None:
        plt.text(i, v + 0.5, f"{v:.2f}%", color='black', ha='center')

plt.show()