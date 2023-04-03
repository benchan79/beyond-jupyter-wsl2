# One prediction per request

The server's code must be in the file `main.py` within a directory called `app`, following FastAPI's guidelines.

## Coding the server (`main.py`)

Begin by importing the necessary dependencies. `pickle` will be used for loading the pre-trained model saved in the `app/wine.pkl` file, `numpy` for tensor manipulation, and the rest for developing the web server with `FastAPI`.

An instance of the `FastAPI` class is created (`app`). This instance will handle all of the functionalities for the server:

```python
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Predicting Wine Class")
```

To represent a data point, create a class from pydantic's `BaseModel` and list each attribute along with its corresponding type.

In this case a data point represents a wine so this class is called `Wine` and all of the features of the model are of type `float`:

```python

# Represents a particular wine (or datapoint)
class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float
```

The classifier is loaded into memory so it can be used for prediction. This can be done in the global scope of the script but here it is done inside a function to show a feature of FastAPI.

If a function is decorated with the `@app.on_event("startup")` decorator, the function is run at the startup of the server. This gives some flexibility if some custom logic needs to be triggered right when the server starts.

The classifier is opened using a context manager and assigned to the `clf` variable, which has to be global so other functions can access it:

```python

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("/app/wine.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)
```

Finally the function that will handle the prediction is created. This function will be run when the `/predict` endpoint of the server is visited. It expects a `Wine` data point.

The information within the `Wine` object is converted into a numpy array of shape `(1, 13)` and then the `predict` method of the classifier is used to make a prediction for the data point. The prediction must be cast into a list using the `tolist` method.

A dictionary is then returned (which FastAPI will convert into `JSON`) containing the prediction.

```python

@app.post("/predict")
def predict(wine: Wine):
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash,
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )

    pred = clf.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}

```

To try the code locally, change the path of the pickle file and use the command `uvicorn main:app --reload` while on the same directory as the `main.py` file.

## Dockerizing the server

Going forward all commands are run within the `wine-classification/` directory.

`main.py` (the server) and its dependencies (`wine.pkl`) are placed in a directory called `app` as explained in the official FastAPI [docs](https://fastapi.tiangolo.com/deployment/docker/) on how to deploy with Docker. The directory structure looks like this:

```text
..
└── no-batch
    ├── app/
    │   ├── main.py (server code)
    │   └── wine.pkl (serialized classifier)
    ├── requirements.txt (Python dependencies)
    ├── wine-examples/ (wine examples to test the server)
    ├── README.md (this file)
    └── Dockerfile
```

## Create the Dockerfile

The `Dockerfile` is made up of all the instructions required to build your image.

### Base Image

```Dockerfile
FROM frolvlad/alpine-miniconda3:python3.7
```

The `FROM` instruction allows the selection of a pre-existing image as the base for the new image. **This means that all of the software available in the base image will also be available.** This is one of Docker's features which allows for reusing images when needed.

In this case the base image is `frolvlad/alpine-miniconda3:python3.7`, let's break it down:

- `frolvlad` is the username of the author of the image.
- `alpine-miniconda3` is its name.
- `python3.7` is the image's tag.

This image contains an [alpine](https://alpinelinux.org/) version of Linux, which is a distribution created to be very small in size. It also includes [miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3. The tag shows that the specific version of Python being used is 3.7. Tagging allows different versions of similar images to be created.

### Installing dependencies

After installing an environment with Python, all of the Python packages that the server will depend on are installed next. First copy the local `requirements.txt` file into the image so it can be accessed by other processes, this can be done via the `COPY` instruction:

```Dockerfile
COPY requirements.txt .
```

Use `pip` to install these Python libraries. To run the command, use the `RUN` instruction:

```Dockerfile
RUN pip install -r requirements.txt && \
  rm requirements.txt
```

The two commands were chained together using the `&&` operator. After installing the libraries specified within `requirements.txt`, it is deleted so the image includes only the necessary files for the server to run.

This can be done using two `RUN` instructions, however, it is a good practice to chain together commands in this manner since Docker creates a new layer every time it encounters a `RUN`, `COPY` or `ADD` instruction. This will result in a bigger image size. For best practices on writing Dockerfiles - [resource](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/).

### Exposing the port

When coding a web server, leave some documentation about the port that the server is going to listen on, using the `EXPOSE` instruction. In this case the server will listen to requests on port 80:

```Dockerfile
EXPOSE 80
```

### Copying your server into the image

To put the code within the image, use the `COPY` instruction to copy the `app` directory within the root of the container:

```Dockerfile
COPY ./app /app
```

### Spinning up the server

This is the command that will be run once a container that uses this image is started. In this case it is the command that will spin up the server by specifying the host and port.

```Dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

Docker uses `JSON` for its configurations and the `CMD` instruction expects the commands as a list that follows `JSON` conventions.

### Putting it all together

```Dockerfile
FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt && \
  rm requirements.txt

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

## Build the image

To build the image, use the `docker build` command.

```bash
docker build -t mlepc4w2-ugl:no-batch .
```

The `-t` flag is used to specify the name of the image and its tag. The tag comes after the colon so in this case the name is `mlepc4w2-ugl` and the tag is `no-batch`.

After a couple of minutes the image should be ready to be used. To see it along with any other images that on the local machine use the `docker images` command. This will display all of the images alongside their names, tags and size.

## Run the container

After the image has been successfully built, run a container from it by using the following command:

```bash
docker run --rm -p 80:80 mlepc4w2-ugl:no-batch
```

- `--rm`: Delete this container after stopping running it. This is to avoid having to manually delete the container. Deleting unused containers helps your system to stay clean and tidy.
- `-p 80:80`: This flags performs an operation knows as port mapping. The container, as well as the local machine, has its own set of ports. To be able to access the port 80 within the container, it needs to be mapped to a port on the computer. In this case it is mapped to the port 80 in the machine.

At the end of the command is the name and tag of the image to be run.

After some seconds the container will start and spin up the server within. FastAPI's logs will also be printed in the terminal.

Go to [localhost:80](http://localhost:80) and there should be a message about the server spinning up correctly.

**Nice work!**

## Make requests to the server

Now that the server is listening to requests on port 80, send `POST` requests to it for predicting classes of wine.

Every request should contain the data that represents a wine in `JSON` format like this:

```json
{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}
```

This example represents a class 1 wine.

FastAPI has a built-in client that allows interaction with the server. The client is located at [localhost:80/docs](http://localhost:80/docs)

`curl` can also be used to send the data directly with the request like this:

```bash
curl -X 'POST' http://localhost/predict \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}'
```

A `JSON` file can be used to avoid typing a long command like this:

```bash
curl -X POST http://localhost:80/predict \
    -d @./wine-examples/1.json \
    -H "Content-Type: application/json"
```

Let's understand the flags used:

- `-X`: Allows you to specify the request type. In this case it is a `POST` request.
- `-d`: Stands for `data` and allows you to attach data to the request.
- `-H`: Stands for `Headers` and it allows you to pass additional information through the request. In this case it is used to the tell the server that the data is sent in a `JSON` format.

There is a directory called `wine-examples` that includes three files, one for each class of wine. Use those to try out the server or pass in some random values.

----
