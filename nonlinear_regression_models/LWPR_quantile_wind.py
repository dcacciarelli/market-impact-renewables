import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
from matplotlib.collections import LineCollection
import matplotlib as mpl

# Set plot aesthetics
mpl.rcParams.update({
    'font.family': 'Helvetica',  # Use Helvetica for clarity
    'axes.labelsize': 18,        # Larger axis labels
    'axes.titlesize': 18,        # Title font size
    'xtick.labelsize': 16,       # X-axis tick label size
    'ytick.labelsize': 16,       # Y-axis tick label size
    'legend.fontsize': 14,       # Legend font size
    # 'axes.grid': True,           # Enable grid
    # 'grid.alpha': 0.2,           # Subtle grid lines
    # 'grid.linestyle': '--',      # Dashed grid lines for a softer look
    'figure.dpi': 300,           # High DPI for high-quality figure
    'savefig.dpi': 300           # High DPI for saving the figure
})


# Define a function to create a colormap with transparency
def get_alpha_colormap(base_cmap, start=0, stop=1, alpha_start=0.1, alpha_stop=0.9):
    # Get the colormap
    colormap = plt.colormaps[base_cmap]
    # Get the color codes from the colormap
    colors = colormap(np.linspace(start, stop, 256))
    # Modify the alpha channel
    alphas = np.linspace(alpha_start, alpha_stop, 256)
    colors[:, 3] = alphas
    # Create a new colormap from the modified color array
    new_cmap = mcolors.ListedColormap(colors)
    return new_cmap


def create_line_segments(x, y, density, color):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Mapping density to alpha between 0.1 (transparent) and 1 (opaque)
    alphas = np.interp(density, [density.min(), density.max()], [0.1, 1.0])
    # Use specified color for all segments
    colors = [(color[0], color[1], color[2], alpha) for alpha in alphas]
    return LineCollection(segments, colors=colors, linewidths=3)


# Function to generate polynomial features up to a given order for two-dimensional input
def polynomial_features(x, order=2):
    n = x.size(0)
    features = [torch.ones(n)]
    if x.shape[1] == 1:
        for i in range(1, order + 1):
            features.append(x[:, 0] ** (i))
    if x.shape[1] == 2:
        for i in range(1, order + 1):
            for j in range(i + 1):
                features.append((x[:, 0] ** (i - j)) * (x[:, 1] ** j))
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


# Tri-cube weight function
def tri_cube(u):
    return torch.where(u < 1, (1 - u ** 3) ** 3, torch.zeros_like(u))


# Perform weighted least squares regression (optimized)
def weighted_least_squares(X, y, weights):
    # Assume X is n x m and weights is n x 1
    weighted_X = X * weights.unsqueeze(1)  # element-wise multiplication to weight each row
    XTWX = weighted_X.T @ X  # This replaces the need for explicitly creating W
    XTWy = weighted_X.T @ y
    beta = torch.linalg.pinv(XTWX) @ XTWy
    return beta.squeeze()


# Calculate weights for all points based on a fitting point and bandwidth
def calculate_weights(data_points, fitting_point, bandwidth):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    scaled_distances = distances / bandwidth
    weights = tri_cube(scaled_distances)
    return weights


# Calculate adaptive bandwidth based on the percentile of distances
def adaptive_bandwidth(data_points, fitting_point, percentile=30):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    return torch.quantile(distances, percentile / 100.0)


# Define the quantile loss function
def quantile_loss(predictions, targets, quantiles, weights):
    assert len(quantiles) == predictions.shape[1], "Each prediction column should correspond to a quantile"
    residuals = targets.unsqueeze(1) - predictions
    losses = torch.max(quantiles * residuals, (quantiles - 1) * residuals)
    weighted_losses = weights.unsqueeze(1) * losses
    return weighted_losses.mean()


# Ensure MPS is available
if torch.backends.mps.is_available():
    print("MPS device is available.")
    mps_device = torch.device("mps")
else:
    print("MPS device not available.")
    # Consider exiting or falling back to CPU processing
    exit()

# Data setup
df = pd.read_csv('/Users/dcac/Data/NESO/day_ahead_wind_portal_preprocessed_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
covariates = ['forecasted_wind_penetration']
response = 'Day Ahead APX Price'
# response='Day-Ahead Nordpool Price'
# response='Within Day Price (MIDP)'
scaler = MinMaxScaler(feature_range=(-1, 1))
df[covariates] = scaler.fit_transform(df[covariates])
features = torch.tensor(df[covariates].values, dtype=torch.float32)
target = torch.tensor(df[response].values, dtype=torch.float32)

X_poly_base = polynomial_features(features)

n_fitting_points = 100
fitting_points = torch.linspace(-1, 1, n_fitting_points).reshape(-1, 1)
# quantiles = torch.tensor([0.1, 0.9])
quantiles = torch.tensor([0.1, 0.9])

# MEAN PREDICTION
coefficients = torch.zeros(len(fitting_points), X_poly_base.size(1), dtype=torch.float32, device=mps_device)
predicted_values = torch.zeros(len(fitting_points), dtype=torch.float32, device=mps_device)

for i, fp in enumerate(tqdm(fitting_points, desc="Fitting points (mean)")):
    bandwidth = adaptive_bandwidth(features, fp, percentile=30)
    weights = calculate_weights(features, fp, bandwidth)
    X_poly = polynomial_features(fp.clone().detach().unsqueeze(0))
    beta = weighted_least_squares(X_poly_base, target, weights)
    coefficients[i] = beta
    predicted_values[i] = torch.matmul(X_poly, beta)

# Transfer results back to CPU if necessary
predicted_values = predicted_values.cpu()
coefficients = coefficients.cpu()

# QUANTILE PREDICTION
coefficients = torch.zeros((len(fitting_points), X_poly_base.size(1), len(quantiles)), dtype=torch.float32)
predicted_quantiles = torch.zeros((len(fitting_points), len(quantiles)), dtype=torch.float32)
n_iterations = 100
for i, fp in enumerate(tqdm(fitting_points, desc="Fitting points (quantiles)")):
    bandwidth = adaptive_bandwidth(features, fp, percentile=30)
    weights = calculate_weights(features, fp, bandwidth)
    beta = torch.randn((X_poly_base.size(1), len(quantiles)), requires_grad=True)
    learning_rate = 10
    # optimizer = torch.optim.SGD([beta], lr=learning_rate)
    optimizer = torch.optim.Adam([beta], lr=learning_rate)

    for epoch in range(n_iterations):
        optimizer.zero_grad()
        predictions = X_poly_base @ beta
        loss = quantile_loss(predictions, target, quantiles, weights)
        loss.backward()
        optimizer.step()
        # learning_rate /= 2
        learning_rate = 0.1 / n_iterations * (epoch + 1) ** (-0.55)

    coefficients[i] = beta.detach()
    X_poly = polynomial_features(fp.clone().detach().unsqueeze(0))
    predicted_quantiles[i] = (X_poly @ beta.detach()).squeeze()

# Plot results
x_values = scaler.inverse_transform(fitting_points.cpu()).flatten()
y_lower = predicted_quantiles.cpu()[:, 0].flatten()
y_upper = predicted_quantiles.cpu()[:, 1].flatten()

# Prepare the figure
x = scaler.inverse_transform(features.cpu()).flatten()
y = target.cpu().flatten()
cmap = get_alpha_colormap('Reds')
hist, bins = np.histogram(x, bins=100, density=True)
density = (hist / hist.max())  # Normalize density to range [0, 1]
x_values = scaler.inverse_transform(fitting_points.cpu()).flatten()
predicted_values_cpu = predicted_values.cpu().flatten()
y_lower = predicted_quantiles.cpu()[:, 0].flatten()
y_upper = predicted_quantiles.cpu()[:, 1].flatten()

# Interpolation of the predicted values for smoother plots
fine_x_values = np.linspace(x_values.min(), x_values.max(), 1000)

# Fit second-order polynomials
poly_mean = np.polyfit(x_values, predicted_values_cpu, 5)
poly_lower = np.polyfit(x_values, y_lower, 5)
poly_upper = np.polyfit(x_values, y_upper, 5)

# Evaluate polynomials on the finer x_values
fine_predicted_values = np.polyval(poly_mean, fine_x_values)
fine_y_lower = np.polyval(poly_lower, fine_x_values)
fine_y_upper = np.polyval(poly_upper, fine_x_values)

hist, bins = np.histogram(x, bins=100, density=True)
density = hist / hist.max()  # Normalize density to range [0, 1]
fine_density = np.interp(fine_x_values, (bins[:-1] + bins[1:]) / 2, density)

# Prepare colormap
cmap = get_alpha_colormap('RdPu')
# cmap = get_alpha_colormap('Blues')
norm = Normalize(vmin=fine_density.min(), vmax=fine_density.max())
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib

fine_x_values = fine_x_values

# Plotting
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
line_mean = create_line_segments(fine_x_values, fine_predicted_values, fine_density,
                                 color=(0.092949, 0.059904, 0.239164, 1.0))
line_lower = create_line_segments(fine_x_values, fine_y_lower, fine_density, color=(0.445163, 0.122724, 0.506901, 1.0))
line_upper = create_line_segments(fine_x_values, fine_y_upper, fine_density, color=(0.445163, 0.122724, 0.506901, 1.0))
ax.add_collection(line_mean)
# ax.add_collection(line_lower)
# ax.add_collection(line_upper)
ax.autoscale()

plt.plot(fine_x_values, fine_predicted_values, c='k', lw=3)
for i in range(len(fine_x_values) - 1):
    x_segment = fine_x_values[i:i + 2]
    y_lower_segment = fine_y_lower[i:i + 2]
    y_upper_segment = fine_y_upper[i:i + 2]
    color = cmap(norm(fine_density[i]))
    ax.fill_between(x_segment, y_lower_segment, y_upper_segment, color=color, alpha=0.2)

# plt.axhline(0, color='gray', linestyle='-', lw=.5)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Data density', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)

ax.tick_params(axis='both', labelsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.2)

# Increase the font size of the colorbar ticks
ax.set_xlabel('Forecasted penetration [%]', fontsize=18, labelpad=10)
ax.set_ylabel('Price [GBP/MWh]', fontsize=18, labelpad=10)

# plt.legend([line_mean, line_lower, line_upper], ['Mean prediction', '10% quantile', '90% quantile'], loc='upper right',
#            fontsize=14)
plt.tight_layout()
plt.savefig(f'/Users/dcac/PycharmProjects/day-ahead-wind-forecast/causal_analysis_wind_solar/plots/quantile_wind.png', format='png', dpi=600)
plt.show()


