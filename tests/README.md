# Tests

To run tests locally:

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
docker-compose up -d
pytest --html=report.html --self-contained-html
```
