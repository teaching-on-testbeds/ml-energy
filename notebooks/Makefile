MARKDOWNS := 00-data.md 01-train.md 02-energy.md
NOTEBOOKS := $(MARKDOWNS:.md=.ipynb)

all: $(NOTEBOOKS)

clean:
	rm -f $(NOTEBOOKS)

%.ipynb:	%.md
	pandoc --wrap=none -i $< -o $@
	

