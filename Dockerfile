FROM ubuntu:cosmic

RUN apt update && apt install -y \
    python3.6 \
    python3-pip \
    python3-grib \
    python3-eccodes \
    locales \
    wget

# Set the locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

RUN apt install -y \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgeos++-dev

RUN mkdir /ccr-submission-code

COPY . /ccr-submission-code

WORKDIR /ccr-submission-code

RUN pip3 install pipenv

RUN pipenv install . --skip-lock
RUN pipenv lock
