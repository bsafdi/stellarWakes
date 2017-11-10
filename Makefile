clean:
	rm -rf build
	rm -f stellarWakes/*.so

build: clean
	python setup.py build_ext --inplace

install:
	python setup.py install
