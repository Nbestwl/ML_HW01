# Authors:
### Casey Sader & Lei Wang

# How to run the program:
### your system needs to support python2.7 before compile
### command to run:
```python
python GMM_Iris.py
python GMM_Sensor.py
```

# Datasets:
1. Iris species.
2. Sensor reading from a wall-following robot(we chose four readings).

# Processes:
1. first the data are read into a dataframe and a histogram is plotted for each feature column individually. From this we determined that two clusters would be the best for our GMM.
2. the inital parameters are chosen randomly, then the new parameters(mean, variance, and pi) are calculated via ri(the probability of the datapoint that it belongs to a cluster class).
3. the experiment runs two iterations, and the program returns nan(not a number) if it runs more than two iterations. We were unable to identify the errors, but we know the variances are getting too small and they are close to zero casuing the program throws "divide by zero" error.
4. in the end of each iteration, the mean and variance are updated.

# Plotting:
1. using the most recent iteration of mean and variance, we calculate gaussian component pdf values at each data point along with the gaussian mixture model as a whole and plot them on top of the histogram.
2. there are issues with the GMM plot as the area under the curve should total to one, this error extends from our calculation and the incorrect update of mean and variance.
3. The plots can be seen in the plot folder.

# Reference:
[reference youtube video](https://www.youtube.com/watch?v=qMTuMa86NzU)
