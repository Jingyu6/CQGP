import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, RationalQuadratic # type: ignore

def gen_point():
	return np.random.rand(1), np.random.rand(1)

cur_x = [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]]
cur_y = [[1], [2], [3], [4], [5], [5]]

gp = GaussianProcessRegressor(
		kernel=ConstantKernel(constant_value=2.0, constant_value_bounds='fixed') 
			* RationalQuadratic(alpha=2.0, length_scale=2.0, alpha_bounds='fixed', length_scale_bounds='fixed') 
			+ WhiteKernel(noise_level=0.1, noise_level_bounds='fixed'),
		normalize_y=True
	)

for _ in range(30):
	x, y = gen_point()
	cur_x.append(x)
	cur_y.append(y)

	gp = gp.fit(cur_x, cur_y)

	x_axis = np.arange(100)
	y_axis, y_std = gp.predict(x_axis.reshape(-1, 1), return_std=True)
	y_axis, y_std = y_axis.flatten(), y_std.flatten()

	plt.scatter(cur_x, cur_y)
	plt.plot(x_axis, y_axis)
	plt.fill_between(x_axis, y_axis - y_std, y_axis + y_std, alpha=0.2)
	plt.show()