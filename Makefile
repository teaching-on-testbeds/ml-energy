all: reserve.ipynb ml_energy.ipynb

clean:
	rm reserve.ipynb ml_energy.ipynb

reserve.ipynb: reserve.md
	pandoc --wrap=none -i reserve.md -o reserve.ipynb


ml_energy.ipynb: notebooks/00-data.md notebooks/01-train.md notebooks/02-energy.md
	pandoc --wrap=none -i notebooks/00-data.md notebooks/01-train.md notebooks/02-energy.md -o ml_energy.ipynb

