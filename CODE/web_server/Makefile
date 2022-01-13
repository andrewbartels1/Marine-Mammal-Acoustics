create-migrations:
	python server/manage.py makemigrations pams

run-migrations:
	python server/manage.py migrate

start-server:
	docker-compose up --build

stop-server:
	docker-compose down

run-server: stop-server start-server
	
process-audio:
	docker-compose run server python audio_processing/DataLoader.py process-data

demo:
	docker-compose run server bash demo.sh