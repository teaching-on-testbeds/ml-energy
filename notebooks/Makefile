MARKDOWNS := 00-data.md 01-train.md 02-energy.md 03-reproduce.md
NOTEBOOKS := $(MARKDOWNS:.md=.ipynb)

all: $(NOTEBOOKS)

clean:
	rm -f $(NOTEBOOKS)

%.ipynb: %.md front-matter.md
	pandoc --wrap=none -i front-matter.md $< -o $@
	

