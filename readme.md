# PCA and Agglomerative Clustering on Iris Dataset

This repository contains the source code and resources for the **PCA and Agglomerative Clustering on the Iris Dataset** project. This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction and Agglomerative Clustering for hierarchical clustering on the Iris dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Imports](#imports)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Modeling](#modeling)
  - [Visualization](#visualization)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **PCA and Agglomerative Clustering on the Iris Dataset** project includes the following key steps:

- **Data Loading**: Importing the Iris dataset from `sklearn`.
- **Data Preprocessing**: Standardizing the data to normalize the feature values.
- **Dimensionality Reduction**: Applying Principal Component Analysis (PCA) to reduce the dataset dimensions from 4 to 2 for easier visualization.
- **Clustering**: Performing Agglomerative Clustering on the reduced dataset to classify the data into clusters.
- **Visualization**: Plotting the dendrogram to visualize the hierarchical clustering and to identify the optimal number of clusters.

## Project Structure

- **notebook.ipynb**: The Jupyter notebook containing the complete code for the analysis, from data loading to clustering and visualization.
- **LICENSE**: The Apache License 2.0 file that governs the use and distribution of this project's code.
- **requirements.txt**: A file listing all the Python libraries and dependencies required to run the project.
- **.gitignore**: A file specifying which files or directories should be ignored by Git.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
```

2. Navigate to the project directory:

```bash 
cd your-repository-name
```

3. Create a virtual environment (optional but recommended):

``` bash
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Jupyter notebook:
``` bash
jupyter notebook notebook.ipynb
```

## Usage

Imports

The notebook begins by importing the necessary libraries:

``` bash 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
%matplotlib inline

```

These libraries are essential for data manipulation (`pandas`, `numpy`), visualization (`seaborn`, `matplotlib`), and clustering (DBSCAN from scikit-learn).

## Data Generation

The synthetic dataset is generated using the `make_moons` function from the `sklearn.datasets` module:

```bash

X, y = make_moons(n_samples=300, noise=0.1)
```
This function creates a dataset with two interleaving half circles, which is useful for demonstrating the effectiveness of clustering algorithms.

## Data Preprocessing
The features are scaled using StandardScaler to normalize the data before applying DBSCAN:

``` bash
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
## Modeling
The DBSCAN algorithm is applied to the scaled data:

``` bash
dbscan = DBSCAN(eps=0.3)
dbscan.fit(X_scaled)
dbscan.labels_

```

This clustering algorithm identifies clusters based on the density of data points, where eps is the maximum distance between two samples for them to be considered as in the same neighborhood.


## Visualization
Though not explicitly included in the notebook, you can visualize the clustering results by plotting the data points and coloring them according to their cluster labels:

``` bash
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan.labels_, cmap='plasma')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering Results')
plt.show()
```

This plot will show how DBSCAN has grouped the data into clusters, with noise points labeled as -1

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.

## Contact
For any inquiries or contributions, feel free to reach out or submit an issue or pull request on GitHub.