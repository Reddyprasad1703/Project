import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics = ['Accuracy', 'Precision', 'F1-Score']
training_scores = [0.92, 0.91, 0.90]
validation_scores = [0.88, 0.85, 0.84]
test_scores = [0.86, 0.83, 0.82]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, training_scores, width, label='Training', color='blue')
plt.bar(x, validation_scores, width, label='Validation', color='orange')
plt.bar(x + width, test_scores, width, label='Test', color='green')

plt.xlabel('Performance Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('RNN Model Performance on Sentiment Analysis (Psychiatric Chatbot)', fontsize=14)
plt.xticks(x, metrics)
plt.ylim(0.7, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()