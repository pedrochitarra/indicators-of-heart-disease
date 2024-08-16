quality_checks:
	echo "Running quality checks"
	isort .

test:
	echo "Running tests"
	pytest --disable-warnings tests/

build: quality_checks test
	echo "Building package"
	docker build . -t pedrochitarra/indicators-of-heart-disease:latest

publish: build
	echo "Publishing package"
	docker push pedrochitarra/indicators-of-heart-disease:latest