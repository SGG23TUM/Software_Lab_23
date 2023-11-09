import json
import os
import glob

# Update the relationship catalog with the new relationship
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

# Mapping of objects to their relationships and the objects they are related to
relationship_rules = {
    "escape sign": ("hanging from", "ceiling"),
    "lighting": ("mounted on", "ceiling"),
    "speaker": ("mounted on", "ceiling"),
    "smoke alarm": ("mounted on", "ceiling"),
    "trash bin": ("placed on", "floor"),
    "elevator button": ("next to", "elevator"),
    "door": ("has opening in", "wall"),
    "staircase": ("equipped with", "handrail"),
    "pipes": ("next to", "wall"),
    "board": ("fixed on", "wall"),
    "fire extinguisher": ("attached to", "wall"),
    "light switch": ("mounted on", "wall"),
    "emergency button": ("mounted on", "wall"),
    "elevator": ("has opening in", "wall")
}

def process_json_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    # Initialize 'relationships' key if it does not exist
    if 'relationships' not in data:
        data['relationships'] = []

    # Map object names to IDs
    object_ids = {obj['names'][0]: obj['object_id'] for obj in data['objects']}

    # Create relationships based on the rules
    for obj in data['objects']:
        obj_name = obj['names'][0]
        if obj_name in relationship_rules:
            predicate, related_obj_name = relationship_rules[obj_name]
            related_obj_id = object_ids.get(related_obj_name)

            # Ensure we have valid IDs before appending
            if related_obj_id is not None:
                data['relationships'].append({
                    "subject_id": obj['object_id'],
                    "object_id": related_obj_id,
                    "predicate": predicate,
                    "relationship_id": predicate_to_idx[predicate]
                })

    # Write the updated data to a new file
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Directory paths
input_directory = 'C:/Users/ataca/OneDrive - TUM/Masaüstü/Software_Lab/Rel_Annotation/annotations'
output_directory = 'C:/Users/ataca/OneDrive - TUM/Masaüstü/Software_Lab/Rel_Annotation/rel_annotations'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each JSON file
for input_file_name in glob.glob(os.path.join(input_directory, '*.json')):
    base_name = os.path.basename(input_file_name)
    output_file_name = os.path.join(output_directory, f"new_{base_name}")
    process_json_file(input_file_name, output_file_name)
    print(f'Processed {input_file_name} and created {output_file_name}')
