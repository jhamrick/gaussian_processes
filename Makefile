PYCMD=python setup.py
GH_PAGES_SOURCES = src docs Makefile

all:
	$(PYCMD) bdist

source:
	$(PYCMD) sdist

upload:
	$(PYCMD) sdist upload

install:
	$(PYCMD) install

clean:
	$(PYCMD) clean --all
	rm -rf dist
	rm -f MANIFEST
	rm -f README
	rm -f *.pyc

test:
	nosetests --nologcapture

gh-pages:
	git checkout gh-pages
	rm -rf build _sources _static
	git checkout master $(GH_PAGES_SOURCES)
	git reset HEAD
	cd docs
	make html
	mv -fv _build/html/* ../
	cd ../
	rm -rf $(GH_PAGES_SOURCES) build
	git add -A
	git ci -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages
	git checkout master
