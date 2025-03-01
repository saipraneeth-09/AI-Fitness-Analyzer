#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# MET values for common exercises
MET_VALUES = {
    "running": 9.8,
    "cycling": 7.5,
    "walking": 3.8,
    "swimming": 8.0,
    "yoga": 3.0
}

def calculate_calories(activity, duration, weight):
    """Calculate calories burned using MET formula."""
    if activity.lower() in MET_VALUES:
        met = MET_VALUES[activity.lower()]
        calories_burned = (met * weight * duration) / 60  # MET formula
        return round(calories_burned, 2)
    else:
        return "Activity not found"


# In[2]:


# Example Data Logging
workouts = [
    {"activity": "running", "duration": 30, "weight": 70},
    {"activity": "cycling", "duration": 45, "weight": 70},
    {"activity": "swimming", "duration": 60, "weight": 70},
]

# DataFrame for visualization
df = pd.DataFrame(workouts)
df["calories_burned"] = df.apply(lambda row: calculate_calories(row["activity"], row["duration"], row["weight"]), axis=1)

# Display the DataFrame
df


# In[3]:


# Plot the data
plt.figure(figsize=(8, 5))
plt.bar(df["activity"], df["calories_burned"], color=["blue", "green", "red"])
plt.xlabel("Activity")
plt.ylabel("Calories Burned")
plt.title("Calories Burned per Activity")
plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script calories-burned.ipynb')

