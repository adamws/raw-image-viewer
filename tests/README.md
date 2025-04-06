# Tests

To run tests locally:

```
python3 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
docker compose up -d
python -m pytest --html=report.html --self-contained-html
```
