import pandas as pd
import numpy as np

def generate_normal_distribution(loc, scale, n):
    return np.random.normal(loc=loc, scale=scale, size=n)

norm_dist = generate_normal_distribution(loc=89, scale=20, n=500)

df = pd.DataFrame(norm_dist)

df.to_csv('player_times.csv')	
