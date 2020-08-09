# realized_volatility
Fair estimation of realized volatility as an indicator for option implied volatility.

- Python 3.6 is used

- Setup virtualenv, at repo root directory
```console
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

- To see the results, execute the following command at repo root directory 
```console
python src/main.py "./data/stockdata3.csv"
```
Execute src/main.py will display the estimated volatility and some auxiliary informaiton for each stock.

- To visualize the data using the notebooks
First, create an ipython kernal using the command:
```console
python -m ipykernel install --user --name realized_volatility
```
Then cd to notebooks directory and launch jupyter notebook by:
```console
jupyter notebook .
```
Choose the kernal named "realized_volatility" just been created previously and run all cells
Plotly extension for Jupyter maybe required.

- For more details please refer to writeup.pdf.
