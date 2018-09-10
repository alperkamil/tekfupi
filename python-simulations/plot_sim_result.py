from glob import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np

experiments = {}  # For different r, threshold and sigma values
name_pattern = r'\D+(\d+)\D+(\d+)\D+(\d+\.\d+)\D+(\d+\.\d+)\D+'
content_pattern = r'\D+(\d+\.\d+)\D+(\d+)\D+(\d+\.\d+)'

# The data must be under the 'results' directory
for file_path in glob('results' + os.sep + '*.txt'):
    name_match = re.match(name_pattern, os.path.basename(file_path))
    r, threshold, sigma, alpha = name_match.group(1), name_match.group(2), name_match.group(3), name_match.group(4)
    file_content = open(file_path).read()
    content_match = re.match(content_pattern, file_content)

    # Create the tuple of results
    alpha, num_congestion, avg_response_time = content_match.group(1), content_match.group(2), content_match.group(3),

    # Add a new experiment as (r, threshold, sigma)
    if (r, threshold, sigma) not in experiments:
        experiments[r, threshold, sigma] = {}
    experiments[r, threshold, sigma][alpha] = num_congestion, avg_response_time

# for experiment in experiments:
experiment = ('2', '64', '0.5')
results = experiments[experiment]
x = []
y = []
for alpha in results:
    x.append(float(alpha))
    y.append(float(results[alpha][1]))

# Sort the data with respect to alpha values
order = np.argsort(x)
xs = np.array(x)[order]
ys = np.array(y)[order]

# Plot and Print
plt.plot(xs, ys)
plt.xlabel(r'$\alpha $', size=15)
plt.ylabel('Avg Response Time (time steps)')
plt.grid()
plt.savefig('r=' + experiment[0] + '_threshold=' + experiment[1] + '_sigma=' + experiment[2] + '.eps', format='eps',
            dpi=1000)
