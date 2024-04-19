all: reserve.ipynb ml_energy.ipynb

clean:
	rm reserve.ipynb ml_energy.ipynb

reserve.ipynb: notebooks/front-matter.md reserve.md
	pandoc --wrap=none -i notebooks/front-matter.md reserve.md -o reserve.ipynb


ml_energy.ipynb: notebooks/front-matter.md notebooks/00-data.md notebooks/01-train.md notebooks/02-energy.md
	pandoc --wrap=none -i notebooks/front-matter.md notebooks/00-data.md notebooks/01-train.md notebooks/02-energy.md -o ml_energy.ipynb

