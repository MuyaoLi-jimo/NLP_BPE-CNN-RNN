import matplotlib.pyplot as plt
import numpy as np


# Example data
data_lengths = np.array([1200, 12000, 120000, 1200000,12000000])
#times = np.array([ , , , , ])
times1_1 = np.array([ 5.602836608886719e-05,0.0002741813659667969 , 0.002363920211791992  ,0.08514189720153809,0.9133858680725098 ])
times1_2 = np.array([ 0.00015997886657714844, 0.0017440319061279297 , 0.009802103042602539, 0.0900571346282959, 0.8816792964935303])
times2_1 = np.array([0.00036716461181640625 , 0.0034689903259277344, 0.03344297409057617, 0.3305060863494873, 3.3577888011932373])
times2_2 = np.array([ 0.00031495094299316406, 0.0028526782989501953, 0.02946305274963379 , 0.2822568416595459, 2.971968173980713])

log_data_lengths = np.log10(data_lengths)
log_times1 = np.log10(times1_1)
log_times2 = np.log10(times1_2)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(log_data_lengths, log_times1, marker='o', label='method 1', linestyle='-', color='blue')
# Plot the second dataset
plt.plot(log_data_lengths, log_times2, marker='s', label='method 2', linestyle='--', color='red')


# Adding labels and title
plt.xlabel('Log(Data Length)')
plt.ylabel('Log(Time)')
plt.title('flatten list Log-Log Plot of Time vs Data Length')

# Annotate each point with its size
for i, txt in enumerate(data_lengths):
    plt.annotate([f"{txt:.0e}",f"{times1_1[i]:.0e}"], (log_data_lengths[i], log_times1[i]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    plt.annotate([f"{txt:.0e}",f"{times1_2[i]:.0e}"], (log_data_lengths[i], log_times2[i]), textcoords="offset points", xytext=(0,-20), ha='center', color='red')

# Adding legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()