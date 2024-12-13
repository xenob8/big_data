# Как воспроизвести решение

## Разархировать данные в каталоге docker volumes

## Запуск эластик и кибаны, заменить волюмы на архивные

```
docker compose up
```
## Запуск монги (бекап)
```commandline
docker run -p 27017:27017  -v MongoVolumePath:/data/db mongo
```

# Запуск парсер
```commandline
python app/api_parser.py
```

# Загрузка в эластик
python -m app/scripts test_preload_elastic_search():

