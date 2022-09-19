# Tests

To run tests locally:

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
docker-compose up -d
INPUTS_PATH=/home/seluser/inputs WEBAPP=webapp pytest --html=report.html --self-contained-html
```
