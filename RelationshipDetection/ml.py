import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


features = []
labels = []

# navigate to the JSON Path
json_folder_path = "datasets/rel_annotations/"

json_files = [os.path.join(json_folder_path, file) for file in os.listdir(json_folder_path) if file.endswith('.json')]


file_names = []
predicate_to_idx = {
    "hanging from": 1,
    "mounted on": 2,
    "behind": 3,
    "in front of": 4,
    "next to": 5,
    "above": 6,
    "under": 7,
    "on": 8,
    "has opening in": 9,
    "standing on": 10,
    "placed on": 11,
    "equipped with": 12,
    "fixed on": 13,
    "attached to": 14
}

idx_to_predicate = {v: k for k, v in predicate_to_idx.items()}

for json_file in json_files:  
    with open(json_file, 'r') as f:
        data = json.load(f)
        for relation in data['relationships']:
            # extraction features
            
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            
            
            subject = next(item for item in data['objects'] if item["object_id"] == subject_id)
            object_ = next(item for item in data['objects'] if item["object_id"] == object_id)
            
            
            feature = [
                subject['x'], subject['y'], subject['w'], subject['h'],
                object_['x'], object_['y'], object_['w'], object_['h']
                
            ]
            features.append(feature)
            
            # add new predicates
            predicate = relation['predicate']

            file_names.append(os.path.basename(json_file))
            labels.append(predicate_to_idx[predicate])  

# Training the Model
features = np.array(features)
labels = np.array(labels)

# Split Training and Testing Dataset
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(features, labels, file_names, test_size=0.2, random_state=42)

# instantiate the model and train it 
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the performance
y_pred = classifier.predict(X_test)



# Accuracy Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix 
cm = confusion_matrix(y_test, y_pred, labels=range(1, 15))

class_names = [idx_to_predicate.get(i, 'Unknown') for i in range(1, 15)]

# Visualize the Confusion matrix
plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.xticks(rotation=45, ha="right")  
plt.yticks(rotation=0)  
plt.tight_layout()  # Present all relationships
plt.show()


# save the Model
joblib.dump(classifier, 'random_forest_classifier.joblib')

# JSON dosyalarını ve ilişkileri bir liste halinde sakla
test_json_files = []
predicted_relationships = []

with open('predictions_with_files.txt', 'w') as f:
    for filename, actual, predicted in zip(file_names_test, y_test, y_pred):
        f.write(f"{filename}: Actual: {idx_to_predicate[actual]}, Predicted: {idx_to_predicate[predicted]}\n")

# Count how many rel. we have
from collections import Counter
file_count = Counter(file_names_test)

# Group the relationships
grouped_predictions = {}
current_index = 0
for file_name, count in file_count.items():
    grouped_predictions[file_name] = y_pred[current_index:current_index + count]
    current_index += count

def update_relationships(json_files, grouped_predictions, idx_to_predicate, json_folder_path):
    for json_file in json_files:
        # read JSON files
        with open(os.path.join(json_folder_path, json_file), 'r') as f:
            data = json.load(f)

        # get the toatl number of relationships in the JSON
        num_relationships = len(data['relationships'])

        # Print out new predicted relationships
        new_relationships = []
        file_predictions = grouped_predictions.get(json_file, [])
        
        
        if len(file_predictions) != num_relationships:
            print(f"Error: The number of predictions for {json_file} does not match the number of relationships.")
            continue  # If total numbers of relationship betweeen actual and predicted rel. not match just pass it for now 

        for i, relation in enumerate(data['relationships']):
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            # add new relations
            new_relationships.append({
                'subject_id': subject_id,
                'object_id': object_id,
                'predicate': idx_to_predicate[file_predictions[i]]
            })

        # Replace old actual rel with predicted ones
        data['relationships'] = new_relationships

        
        with open(os.path.join(json_folder_path, json_file.replace('.json', '_new_relationships.json')), 'w') as f:
            json.dump(data, f, indent=4)


update_relationships(file_names_test, grouped_predictions, idx_to_predicate, "datasets/rel_annotations/")