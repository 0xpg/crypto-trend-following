PYTHON := $(or $(PYTHON),python3)

.PHONY: all funding panel backtest ccxt-backtest app claims report test clean

all: panel backtest claims report

funding:
	$(PYTHON) scripts/fetch_funding.py

panel:
	$(PYTHON) scripts/build_panel.py

backtest:
	$(PYTHON) scripts/backtest.py

ccxt-backtest:
	$(PYTHON) scripts/ccxt_backtest.py

app:
	$(PYTHON) -m streamlit run app.py

claims:
	$(PYTHON) scripts/claims.py

report:
	$(PYTHON) scripts/figures.py
	$(PYTHON) scripts/report.py

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf results/*.csv results/*.json results/*.parquet results/figures/*.png results/report.md
