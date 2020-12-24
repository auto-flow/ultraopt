test:
	pytest tests

reinstall:
	pip uninstall ultraopt -y
	rm -rf build dist *.egg-info
	python setup.py install

upload:
	rm -rf build dist *.egg-info
	python setup.py sdist
	twine upload dist/*

clean:
	rm -rf build dist *.egg-info
