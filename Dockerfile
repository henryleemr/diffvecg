# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Change the timezone in the container
ENV TZ=Asia/Kuala_Lumpur
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install requirements
RUN pip3 install -r requirements.txt

# # Install production dependencies.
# RUN pip install Flask gunicorn

# # Run the web service on container startup. Here we use the gunicorn
# # webserver, with one worker process and 8 threads.
# # For environments with multiple CPU cores, increase the number of workers
# # to be equal to the cores available.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app


# Get c++ compiler
# https://stackoverflow.com/questions/29732990/installing-a-gcc-compiler-onto-a-docker-container
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get clean
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

# https://github.com/emscripten-core/emscripten/issues/9669
RUN apt-get update && apt-get install make

# https://stackoverflow.com/questions/31421327/cmake-cxx-compiler-broken-while-compiling-with-cmake
RUN apt-get install -y build-essential


# Install mp3 creator
RUN apt install ffmpeg

# Install
CMD python setup.py install
