# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from kanly.api import build_data_model

# num_periods = 52*3
# num_channels = 6

# period = np.arange(num_periods)
# seasonality = np.sin(2 * np.pi * np.arange(num_periods) / 52)
# trend = 4 * np.arange(num_periods) / num_periods

# revenue = 4 + trend + .5 * seasonality + .2 * np.random.randn(num_periods)

# df = pd.DataFrame({
#     'revenue': revenue, 
#     'time_linear': np.arange(num_periods),
#     'year': period // 52,
#     'week_of_year': period % 52,
# })

# plt.plot(df['revenue'])
# plt.show()

# data_string = """
# self.revenue = `revenue`
# self.period = `period`
# self.channel = `channel`
# """

# model = build_data_model(data_string, model_string, df)