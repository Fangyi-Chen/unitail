all:
    # install pycocotools locally
	python setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	python setup.py build_ext install
	rm -rf build

clean:
	rm -rf pycocotools_q/*.so
	rm -rf pycocotools_q/poly_nms_gpu/*.so