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

cp_img:
	cp -r experiments/synthetic/*.png ../ultraopt_img  && \
 	cp -r experiments/tabular_benchmarks/*.png ../ultraopt_img && \
 	cd ../ultraopt_img && \
 	git add . && \
 	git commit -m "update" && \
 	git push origin master

print_img_url:
	sh shell_utils/print_img_url.sh

build_zh_docs:
	mkdir -p docs/build/zh && \
	cd docs/zh_CN && \
	make html && \
	cp -rf build/html/* ../build/zh

