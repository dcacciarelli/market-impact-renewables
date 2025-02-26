# Market Impact of Renewables on Electricity Prices

## Overview
This repository contains the code and data analysis related to the paper **"Do we actually understand the impact of renewables on electricity prices? A causal inference approach"**. The study investigates the causal effect of wind and solar penetration on electricity market prices using **Double Machine Learning (DML)** techniques.

## Key Findings
- Increased wind and solar penetration can significantly impact within-day and day-ahead electricity market prices.
- **Causal Inference with DML** helps isolate the true effect of renewables while controlling for confounders.
- A **sliding window analysis** is employed to study how the effect varies across different penetration levels.

## Repository Structure
```
📂 market-impact-renewables
│── 📄 README.md        # This file
│── 📄 manuscript.pdf   # Paper explaining methodology and findings
│── 📂 code
│   ├── DML_overall_solar.py   # Main analysis script for solar penetration
│   ├── DML_overall_wind.py    # Main analysis script for wind penetration
│   ├── DML_utils_solar.py    # Utility functions for solar analysis
│   ├── DML_utils_wind.py     # Utility functions for wind analysis
│   ├── dml_vs_mean_plot_solar.py  # Plot CATE vs mean prices for solar
│   ├── dml_vs_mean_plot_wind.py   # Plot CATE vs mean prices for wind
│── 📂 data
│   ├── processed_data.csv  # Preprocessed electricity market data
│   ├── results_solar.csv   # Results from solar analysis
│   ├── results_wind.csv    # Results from wind analysis
```

## Getting Started
### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib tqdm lightgbm scikit-learn scipy
```

### Running the Analysis
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/market-impact-renewables.git
   cd market-impact-renewables
   ```
2. Run the solar market impact analysis:
   ```bash
   python code/DML_overall_solar.py
   ```
3. Run the wind market impact analysis:
   ```bash
   python code/DML_overall_wind.py
   ```

## Visualizations
### Causal Effect of Solar Penetration on Prices
<img src='mean_vs_cate_solar.png' width='500'/>

### Causal Effect of Wind Penetration on Prices
<img src='mean_vs_cate_wind.png' width='500'/>

## Contact
For questions or collaborations, reach out to **d.cacciarelli@gmail.com** or connect on [LinkedIn](https://www.linkedin.com/in/cacciarelli/).
