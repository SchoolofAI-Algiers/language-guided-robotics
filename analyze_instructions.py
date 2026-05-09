import pandas as pd
import numpy as np

# Load instructions CSV
df = pd.read_csv('nlp/nlp_instructions_125.csv')
print("DataFrame shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 30 rows:")
print(df.to_string())

# Analyze instruction types
if 'instruction' in df.columns:
    instructions = df['instruction'].tolist()
    print(f"\n\nTotal instructions: {len(instructions)}")
    
    # Extract task types
    task_types = set()
    keywords = {
        'reach': ['reach', 'approach', 'go to', 'move to'],
        'pick': ['pick', 'grasp', 'grab', 'pickup'],
        'lift': ['lift', 'raise', 'up', 'elevate'],
        'place': ['place', 'put', 'drop', 'set down'],
        'push': ['push', 'shove', 'slide'],
        'pull': ['pull', 'drag', 'tug'],
        'lower': ['lower', 'down', 'descend']
    }
    
    task_distribution = {k: 0 for k in keywords.keys()}
    
    for inst in instructions:
        inst_lower = inst.lower()
        for task, kws in keywords.items():
            if any(kw in inst_lower for kw in kws):
                task_distribution[task] += 1
                break
    
    print("\nTask distribution:")
    for task, count in sorted(task_distribution.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")
