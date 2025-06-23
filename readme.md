# README

This python-project estimates GNSS-R derived soil moisture by downloading, processing and analyzing satellite data from different data sources (CYGNSS, SMAP, ERA5-Land, ISMN). The goal is to produce accurate and high-resolution soil moisture estimates over selected study areas.


## Setting up the project

Setting up the project is pretty straight forward, but a couple steps are required:

### Cloning the repository

In a terminal on your computer navigate to the path where you want to store the project locally on your computer. Then clone the project using HTTPS:

```bash
git clone https://github.com/AndreasMaakes/soil-moisture-monitoring-cygnss-era5-smap.git
```

### Setting up the virtual enviroment

Through the terminal or an IDE, navigate to the root folder of the project (replace the path below with your local path):

```bash
cd ~/Desktop/my_project
```

Then create the virtual enviroment:

```bash
python -m venv venv
```

Activate the environment:

For macOS:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Installing the dependencies

When the user is inside the venv and wants to install the dependencies for the project, they have to use these commands:

```bash
pip install -r requirements.txt
```

You should now have the necessary packs installed to run the scripts.


## Project Structure

The project is organized into the following main folders:

### `data/`
Contains all input data files grouped by source:

- `CYGNSS/` – GNSS Reflectometry-based data  
- `SMAP/` – Passive microwave satellite soil moisture data from NASA  
- `ERA5/` – Reanalysis data from ECMWF  
- `ISMN/` – In-situ soil moisture observations  
- `Timeseries/` – Downloaded and preprocessed time series datasets  

---

### `dataHandling/`
Scripts for loading, preprocessing, and organizing data:

- `main.py` – Main script for coordinating data handling  
- `data_fetching_time_series.py` – Downloads and organizes time series data  
- `create_dates_array.py` – Generates date arrays for time-based processing  
- `__init__.py` – Makes the folder a Python module  
- Subfolders:
  - `CYGNSS/` – CYGNSS-specific data handling  
  - `ERA5/` – ERA5-specific data handling  
  - `SMAP/` – SMAP-specific data handling  

---

### `plotting/`
Scripts related to visualization and analysis:

- `plot_timeseries.py` – Plotting of time series from different datasets  
- `plot_correlation.py` – Visualizing correlations between soil moisture sources  
- `plot_incidence_angle.py` – Plots incidence angle vs soil moisture (e.g. for CYGNSS)  
- `timeseries_correlation.py` – Time series cross-source correlation plots  
- `timeseries_correlation_ismn.py` – Correlation plots with in-situ data  
- `plotting.py` – General plotting functions  
- `__init__.py` – Makes the folder a Python module  
- Subfolders:
  - `CYGNSS/`, `ERA5/`, `ISMN/`, `SMAP/` – Dataset-specific plot utilities  

---

Each script is modular and meant to be used in a pipeline from data fetching to analysis and plotting.

## Downloading data
