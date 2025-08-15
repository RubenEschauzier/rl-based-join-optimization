import json
import matplotlib.pyplot as plt

# Path to your JSON file
file_location = 'failed_execution_times.json'
file_location_not_failed = 'not_failed_execution_times.json'
# Load the list from the JSON file
with open(file_location, 'r') as f:
    data = json.load(f)

with open(file_location_not_failed, 'r') as f:
    data_not = json.load(f)
    for i in range(20):
        data_not.remove(max(data_not))

# Plot the histogram
plt.hist(data, bins=20, edgecolor='black')
plt.xlabel('Execution time')
plt.ylabel('Frequency')
plt.title('Failed query execution times')
plt.show()


plt.hist(data_not, bins=50, edgecolor='black')
plt.xlabel('Execution time')
plt.ylabel('Frequency')
plt.title('Completed query execution times')
plt.show()

