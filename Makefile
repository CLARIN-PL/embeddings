build_docker:
	docker build -t clarin/embeddings:local .
test:
	docker run -ti --rm -v `pwd`:/code  clarin/embeddings:local -- 'poetry run poe test'
check:
	docker run -ti --rm -v `pwd`:/code  clarin/embeddings:local -- 'poetry run poe check'
