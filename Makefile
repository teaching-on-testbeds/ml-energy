all: reserve.ipynb

clean:
	rm reserve.ipynb

reserve.ipynb: reserve.md
	pandoc --wrap=none -i reserve.md -o reserve.ipynb
